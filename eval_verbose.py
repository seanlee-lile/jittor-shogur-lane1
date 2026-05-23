# eval_verbose.py
"""
为每个数据集输出详细的候选预测信息：
- expected: 真实目标节点
- predicted: 模型排序第一的候选
- prob: 对应概率
同时计算并打印 MRR，并将详情保存为 data/<dataset>/<dataset>_verbose.csv

用法示例：
    conda activate jittor
    python eval_verbose.py --data_dir ./data
"""
import os
import os.path as osp
import re
import sys
import argparse
import numpy as np
import pandas as pd

# ensure local package import
root = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, osp.join(root, 'JittorGeometric'))

import jittor as jt
from jittor_geometric.data import TemporalData
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
from jittor_geometric.nn.models.craft import CRAFT

jt.flags.use_cuda = 1

# 寻找并加载预测分数文件（与 eval_ranking 保持一致）

def find_datasets(data_dir):
    out = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path):
            continue
        if name.isdigit() or re.match(r'^dataset\d+$', name):
            out.append(name)
    return out


def compute_mrr_and_verbose(scores, test_df):
    # scores: numpy (N, C)
    # test_df: pandas DataFrame, candidates 从第3列开始
    if 'dst' in test_df.columns:
        true = test_df['dst'].values
    else:
        true = test_df.iloc[:, 2].values
    cands = test_df.iloc[:, 2:].values

    n = len(true)
    rr_sum = 0.0

    rows = []
    for i in range(n):
        row_scores = scores[i]
        order = np.argsort(-row_scores)
        ranked = cands[i][order]
        probs = row_scores[order]
        predicted = ranked[0]
        predicted_prob = probs[0]
        # find true position
        pos_arr = np.where(ranked == true[i])[0]
        if pos_arr.size > 0:
            pos = pos_arr[0]
            rr_sum += 1.0 / (pos + 1)
        else:
            pos = None
        rows.append({
            'expected': int(true[i]),
            'predicted': int(predicted),
            'predicted_prob': float(predicted_prob),
            'rank_of_expected': int(pos + 1) if pos is not None else -1
        })

    mrr = rr_sum / n
    return mrr, pd.DataFrame(rows)


def eval_dataset(data_dir, dataset):
    pred_path = osp.join(data_dir, dataset, f"{dataset}_result.csv")
    test_csv = osp.join(data_dir, dataset, "test.csv")
    # if result file missing but model exists, try to generate predictions from pkl
    if not osp.exists(test_csv):
        return None
    if not osp.exists(pred_path):
        # try to find model checkpoint
        model_pkl = osp.join('saved_models', f"{dataset}_CRAFT_best.pkl")
        if osp.exists(model_pkl):
            print(f'  result file not found, generating from model {model_pkl}...')
            ok = generate_results_from_pkl(data_dir, dataset, model_pkl)
            if not ok:
                print(f'  failed to generate results for {dataset}')
                return None
        else:
            return None

    scores = np.loadtxt(pred_path, delimiter=',')
    test_df = pd.read_csv(test_csv)
    if scores.ndim == 1:
        scores = scores.reshape((scores.shape[0], -1))
    mrr, verbose_df = compute_mrr_and_verbose(scores, test_df)
    out_csv = osp.join(data_dir, dataset, f"{dataset}_verbose.csv")
    verbose_df.to_csv(out_csv, index=False)
    return mrr, out_csv


def generate_results_from_pkl(data_dir, dataset, pkl_path, batch_size=200, num_neighbors=30):
    # load train to build neighbor sampler and node stats
    train_csv = osp.join(data_dir, dataset, 'train.csv')
    test_csv = osp.join(data_dir, dataset, 'test.csv')
    if not osp.exists(train_csv) or not osp.exists(test_csv):
        return False
    df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    src_np = df['src'].values.astype(np.int32)
    dst_np = df['dst'].values.astype(np.int32)
    t_np = df['time'].values.astype(np.int32)
    edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

    full_data = TemporalData(
        src=jt.Var(src_np), dst=jt.Var(dst_np), t=jt.Var(t_np), edge_ids=jt.Var(edge_ids_np)
    )
    full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=1)

    # prepare model
    max_node = max(int(src_np.max()), int(dst_np.max()), int(test_df.iloc[:, 2:].values.max()))
    node_size = max_node + 1
    dst_min = min(int(dst_np.min()), int(test_df.iloc[:, 2:].values.min()))
    src_min = int(src_np.min())

    model = CRAFT(
        n_layers=2, n_heads=2, hidden_size=64, hidden_dropout_prob=0.1, attn_dropout_prob=0.1,
        hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02, n_nodes=node_size,
        max_seq_length=num_neighbors, loss_type='BPR', use_pos=True, input_cat_time_intervals=False,
        output_cat_time_intervals=True, output_cat_repeat_times=True, num_output_layer=1,
        emb_dropout_prob=0.1, skip_connection=True
    )
    try:
        model.load_state_dict(jt.load(pkl_path))
    except Exception as e:
        print(f'  error loading model: {e}')
        return False
    model.set_min_idx(src_min, dst_min)
    model.eval()

    # prepare test arrays
    test_src = test_df['src'].values.astype(np.int32)
    test_time = test_df['time'].values.astype(np.int32)
    test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

    # generate scores in batches
    num_samples = len(test_src)
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_scores = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)
        batch_src = test_src[start:end]
        batch_time = test_time[start:end]
        batch_cand = test_candidates[start:end]

        src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=batch_src, node_interact_times=batch_time, num_neighbors=num_neighbors)
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        test_dst = jt.Var(batch_cand)

        dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=test_dst.flatten().numpy(),
            node_interact_times=np.broadcast_to(batch_time[:, np.newaxis], (len(batch_time), test_dst.shape[1])).flatten(),
            num_neighbors=1)
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

        logits = model.forward(src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times),
                              jt.Var(batch_time), test_dst=test_dst_adj, dst_last_update_times=dst_last_update_time)
        probs = jt.sigmoid(logits.squeeze(-1)).numpy()
        all_scores.append(probs)

    scores = np.vstack(all_scores)
    out_path = osp.join(data_dir, dataset, f"{dataset}_result.csv")
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for row in scores:
            f.write(','.join([f'{p:.8f}' for p in row]) + '\n')
    print(f'  saved generated results to {out_path}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    datasets = find_datasets(args.data_dir)
    if not datasets:
        print('No dataset folders found in', args.data_dir)
        return

    for ds in datasets:
        print(f'Evaluating {ds}...')
        res = eval_dataset(args.data_dir, ds)
        if res is None:
            print('  Skipping', ds, '(missing files)')
            continue
        mrr, out_csv = res
        print(f'  MRR: {mrr:.6f}; verbose saved to {out_csv}')


if __name__ == '__main__':
    main()

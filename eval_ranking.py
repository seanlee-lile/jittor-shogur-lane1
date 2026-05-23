# eval_ranking.py
import os
import re
import argparse
import numpy as np
import pandas as pd


def ranking_mrr(scores, cands, true):
    num = len(true)
    rr_sum = 0.0
    for i in range(num):
        row_scores = scores[i]
        order = np.argsort(-row_scores)
        ranked_cands = cands[i][order]
        # 找到真实目标位置
        try:
            pos = np.where(ranked_cands == true[i])[0][0]
            rr_sum += 1.0 / (pos + 1)
        except IndexError:
            # 真实目标不在候选中，记为 0
            continue
    mrr = rr_sum / num
    return mrr


def eval_dataset(data_dir, dataset):
    pred_path = os.path.join(data_dir, dataset, f"{dataset}_result.csv")
    test_csv = os.path.join(data_dir, dataset, "test.csv")
    if not os.path.exists(pred_path):
        print(f"  Skipping {dataset}: result file not found: {pred_path}")
        return None
    if not os.path.exists(test_csv):
        print(f"  Skipping {dataset}: test file not found: {test_csv}")
        return None

    scores = np.loadtxt(pred_path, delimiter=',')
    test_df = pd.read_csv(test_csv)

    if 'dst' in test_df.columns:
        true = test_df['dst'].values
    else:
        true = test_df.iloc[:, 2].values

    cands = test_df.iloc[:, 2:].values
    if scores.ndim == 1:
        scores = scores.reshape((scores.shape[0], -1))

    mrr = ranking_mrr(scores, cands, true)
    return mrr


def find_datasets(data_dir):
    out = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path):
            continue
        # 包含两类：纯数字目录，或 dataset+数字（如 dataset1）
        if name.isdigit() or re.match(r'^dataset.*', name):
            out.append(name)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()

    datasets = find_datasets(args.data_dir)
    if not datasets:
        print('No numeric or datasetN folders found in', args.data_dir)
        return

    results = []
    for ds in datasets:
        print(f'Evaluating {ds}...')
        mrr = eval_dataset(args.data_dir, ds)
        if mrr is None:
            continue
        print(f'  MRR: {mrr:.6f}')
        results.append({'dataset': ds, 'mrr': mrr})

    # summary saving removed by user request


if __name__ == '__main__':
    main()
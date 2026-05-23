# eval_loss.py
import os, argparse, numpy as np, pandas as pd
import os.path as osp
import sys
# Ensure local JittorGeometric package is importable
root = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, osp.join(root, 'JittorGeometric'))
import jittor as jt
from jittor_geometric.data import TemporalData
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
from jittor_geometric.nn.models.craft import CRAFT

jt.flags.use_cuda = 1
print('EVAL_LOSS_SCRIPT_STARTED')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--save_dir', default='./saved_models')
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--num_batches', type=int, default=10, help='compute loss on first N batches')
args = parser.parse_args()

# load test/val split like main.py (这里以 val 为例)
df = pd.read_csv(f'{args.data_dir}/{args.dataset}/train.csv')
num_total = len(df)
num_val = int(num_total * 0.15)
num_train = num_total - num_val

src_np = df['src'].values.astype(np.int32)
dst_np = df['dst'].values.astype(np.int32)
t_np = df['time'].values.astype(np.int32)
edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

val_data = TemporalData(
    src=jt.Var(src_np[num_train:]),
    dst=jt.Var(dst_np[num_train:]),
    t=jt.Var(t_np[num_train:]),
    edge_ids=jt.Var(edge_ids_np[num_train:])
)
val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=1.0)

full_data = TemporalData(src=jt.Var(src_np), dst=jt.Var(dst_np), t=jt.Var(t_np), edge_ids=jt.Var(edge_ids_np))
full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=1)

# model init MUST match how it was created when saved
max_node = max(int(src_np.max()), int(dst_np.max()))
node_size = max_node + 1
num_neighbors = 30
model = CRAFT(
    n_layers=2, n_heads=2, hidden_size=64, hidden_dropout_prob=0.1, attn_dropout_prob=0.1,
    hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02, n_nodes=node_size,
    max_seq_length=num_neighbors, loss_type='BPR', use_pos=True, input_cat_time_intervals=False,
    output_cat_time_intervals=True, output_cat_repeat_times=True, num_output_layer=1,
    emb_dropout_prob=0.1, skip_connection=True
)

best_path = os.path.join(args.save_dir, f'{args.dataset}_CRAFT_best.pkl')
if not os.path.exists(best_path):
    raise SystemExit(f'Best model not found: {best_path}')
model.load_state_dict(jt.load(best_path))
model.set_min_idx(int(src_np.min()), int(dst_np.min()))
model.eval()

losses = []
for i, batch in enumerate(val_loader):
    if i >= args.num_batches:
        break
    src = jt.array(batch.src); dst = jt.array(batch.dst); t = jt.array(batch.t); neg_dst = jt.array(batch.neg_dst)
    src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
        node_ids=src.numpy(), node_interact_times=t.numpy(), num_neighbors=num_neighbors)
    neighbor_num = (src_neighb_seq != 0).sum(axis=1)
    if neighbor_num.sum() == 0:
        continue
    pos_item = jt.Var(dst).unsqueeze(-1)
    neg_item = jt.Var(neg_dst).unsqueeze(-1)
    test_dst = jt.cat([pos_item, neg_item], dim=-1)
    dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
        node_ids=test_dst.flatten().numpy(),
        node_interact_times=np.broadcast_to(t.numpy()[:,None], (len(t), test_dst.shape[1])).flatten(),
        num_neighbors=1)
    dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
    dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000
    dst_last_update_time = jt.Var(dst_last_update_time)

    loss, pos_score, neg_score = model.calculate_loss(
        src_neighb_seq=jt.Var(src_neighb_seq),
        src_neighb_seq_len=jt.Var(neighbor_num),
        src_neighb_interact_times=jt.Var(src_neighb_interact_times),
        cur_pred_times=jt.Var(t),
        test_dst=test_dst,
        dst_last_update_times=dst_last_update_time)
    losses.append(loss.item())

print('Avg loss over', len(losses), 'batches:', np.mean(losses))
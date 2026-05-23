"""
CRAFT 模型训练和预测脚本

此脚本使用 JittorGeometric 库实现 CRAFT (Continuous-Time Relational Attention with Fourier Transform) 模型，
用于时序图上的链路预测任务。主要功能包括：
1. 加载时序图数据
2. 训练 CRAFT 模型进行链路预测
3. 使用训练好的模型生成测试集预测分数

支持早停机制和负采样。

使用方法：
python main.py --dataset <dataset_name> --data_dir <data_dir> --save_dir <save_dir>

参数说明：
- dataset: 数据集名称 (必需)
- data_dir: 数据目录 (默认: ./data)
- save_dir: 模型保存目录 (默认: ./saved_models)
- output_dir: 输出目录 (默认: 与 data_dir 相同)
- epochs: 训练轮数 (默认: 100)
- batch_size: 批次大小 (默认: 200)
- early_stop: 早停耐心值 (默认: 10)
"""
import os
import os.path as osp
import sys

# 将 JittorGeometric 添加到 Python 路径中，使其可以被导入
root = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, osp.join(root, 'JittorGeometric'))

# 设置 Jittor 同步模式，确保操作同步执行
os.environ['JT_SYNC'] = '1'

# 导入必要的库
import jittor as jt  # Jittor 深度学习框架
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
from tqdm import tqdm  # 进度条显示
from sklearn.metrics import average_precision_score, roc_auc_score  # 评估指标
from jittor_geometric.data import TemporalData  # 时序图数据结构
from jittor_geometric.nn.models.craft import CRAFT  # CRAFT 模型
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler  # 数据加载器和邻居采样器
import argparse  # 命令行参数解析

# 启用 CUDA 支持，使用 GPU 加速
jt.flags.use_cuda = 1


# 验证函数：计算模型在验证集上的平均精度 (AP) 和 AUC 分数
def test_val(model, loader, full_neighbor_sampler, num_neighbors):
    model.eval()  # 设置模型为评估模式
    ap_list, auc_list = [], []  # 存储每个批次的 AP 和 AUC
    loader_tqdm = tqdm(loader, ncols=120, desc='Validation')  # 进度条
    for _, batch_data in enumerate(loader_tqdm):
        # 从批次数据中提取源节点、目标节点、时间戳和负采样目标节点
        src = jt.array(batch_data.src)
        dst = jt.array(batch_data.dst)
        t = jt.array(batch_data.t)
        neg_dst = jt.array(batch_data.neg_dst)

        # 获取源节点的邻居序列，用于模型输入
        src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=src.numpy(), node_interact_times=t.numpy(), num_neighbors=num_neighbors)
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)  # 计算每个节点的邻居数量

        # 准备正样本和负样本的目标节点
        pos_item = jt.Var(dst).unsqueeze(1)
        neg_item = jt.Var(neg_dst).unsqueeze(1)
        test_dst = jt.cat([pos_item, neg_item], dim=1)  # 拼接正负样本

        # 获取目标节点的最后邻居和更新时间
        dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=test_dst.flatten().numpy(),
            node_interact_times=np.broadcast_to(t.numpy()[:,np.newaxis], (len(t), test_dst.shape[1])).flatten(),
            num_neighbors=1)
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000  # 处理无邻居的情况
        dst_last_update_time = jt.Var(dst_last_update_time)

        # 使用模型进行预测
        pos_score, neg_score = model.predict(
            src_neighb_seq=jt.Var(src_neighb_seq),
            src_neighb_seq_len=jt.Var(neighbor_num),
            src_neighb_interact_times=jt.Var(src_neighb_interact_times),
            cur_pred_times=jt.Var(t),
            test_dst=test_dst,
            dst_last_update_times=dst_last_update_time)

        # 计算评估指标
        y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])  # 真实标签
        y_score = np.concatenate([pos_score, neg_score.flatten()])  # 预测分数
        ap_list.append(average_precision_score(y_true, y_score))
        auc_list.append(roc_auc_score(y_true, y_score))

    return {'AP': np.mean(ap_list), 'AUC': np.mean(auc_list)}  # 返回平均 AP 和 AUC

def train(model, optimizer, train_loader, val_loader, full_neighbor_sampler, num_neighbors, num_epochs, save_path, dataset_name, early_stop_patience=10):
    # 训练函数：训练模型并实现早停机制
    best_ap = 0  # 最佳验证 AP
    patience_counter = 0  # 早停计数器

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        train_losses = []  # 存储训练损失
        train_tqdm = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch+1}')  # 训练进度条

        for batch_idx, batch_data in enumerate(train_tqdm):
            # 提取批次数据
            src = jt.array(batch_data.src)
            dst = jt.array(batch_data.dst)
            t = jt.array(batch_data.t)
            neg_dst = jt.array(batch_data.neg_dst)

            # 获取邻居信息
            src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=src.numpy(), node_interact_times=t.numpy(), num_neighbors=num_neighbors)
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)

            if neighbor_num.sum() == 0:  # 跳过无邻居的批次
                continue

            # 准备训练数据
            pos_item = jt.Var(dst).unsqueeze(-1)
            neg_item = jt.Var(neg_dst).unsqueeze(-1)
            test_dst = jt.cat([pos_item, neg_item], dim=-1)

            # 获取目标节点信息
            dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=test_dst.flatten().numpy(),
                node_interact_times=np.broadcast_to(t.numpy()[:,np.newaxis], (len(t), test_dst.shape[1])).flatten(),
                num_neighbors=1)
            dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
            dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000
            dst_last_update_time = jt.Var(dst_last_update_time)

            # 计算损失
            loss, _, _ = model.calculate_loss(
                src_neighb_seq=jt.Var(src_neighb_seq),
                src_neighb_seq_len=jt.Var(neighbor_num),
                src_neighb_interact_times=jt.Var(src_neighb_interact_times),
                cur_pred_times=jt.Var(t),
                test_dst=test_dst,
                dst_last_update_times=dst_last_update_time)

            # 反向传播和优化
            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()  # 同步所有操作
            train_losses.append(loss.item())
            train_tqdm.set_description(f'Epoch {epoch+1}, loss: {loss.item():.4f}')

        # 打印训练损失
        print(f'Epoch {epoch+1}, Train Loss: {np.mean(train_losses):.4f}')
        # 验证
        val_res = test_val(model, val_loader, full_neighbor_sampler, num_neighbors)
        print(f'Epoch {epoch+1}, Val: {val_res}')

        # 早停检查
        current_ap = val_res['AP']
        if current_ap > best_ap:
            best_ap = current_ap
            patience_counter = 0
            jt.save(model.state_dict(), f'{save_path}/{dataset_name}_CRAFT_best.pkl')  # 保存最佳模型
            print(f'  -> New best AP: {best_ap:.6f}, model saved!')
        else:
            patience_counter += 1
            print(f'  -> No improvement for {patience_counter} epoch(s), best AP: {best_ap:.6f}')

        # 保存最新模型
        jt.save(model.state_dict(), f'{save_path}/{dataset_name}_CRAFT.pkl')

        if patience_counter >= early_stop_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs!')
            print(f'Best validation AP: {best_ap:.6f}')
            break

    return best_ap

    return best_ap

# 测试函数：使用训练好的模型生成测试集的预测分数
def test_competition(model, test_src, test_time, test_candidates, full_neighbor_sampler, num_neighbors, batch_size=200):
    model.eval()
    all_scores = []
    num_samples = len(test_src)
    num_batches = (num_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches), ncols=120, desc='Testing')
    for batch_idx in pbar:
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
            node_interact_times=np.broadcast_to(batch_time[:,np.newaxis], (len(batch_time), test_dst.shape[1])).flatten(),
            num_neighbors=1)
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

        logits = model.forward(src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times),
                              jt.Var(batch_time), test_dst=test_dst_adj, dst_last_update_times=dst_last_update_time)
        probs = jt.sigmoid(logits.squeeze(-1)).numpy()
        all_scores.append(probs)

    return np.vstack(all_scores)

# 主函数：解析命令行参数并执行训练和预测流程
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='模型保存目录')
parser.add_argument('--output_dir', type=str, default=None, help='输出目录 (默认与数据目录相同)')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=200, help='批次大小')
parser.add_argument('--early_stop', type=int, default=10, help='早停耐心值')
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

print('='*80)
print(f'CRAFT Competition - Dataset: {args.dataset}')
print('='*80)

# 加载数据 - 使用 int32 类型像 JODIE 一样
df = pd.read_csv(f'{args.data_dir}/{args.dataset}/train.csv')  # 训练数据
test_df = pd.read_csv(f'{args.data_dir}/{args.dataset}/test.csv')  # 测试数据

# 提取数据列
src_np = df['src'].values.astype(np.int32)  # 源节点
dst_np = df['dst'].values.astype(np.int32)  # 目标节点
t_np = df['time'].values.astype(np.int32)  # 时间戳
edge_ids_np = np.arange(len(df), dtype=np.int32) + 1  # 边 ID

test_src = test_df['src'].values.astype(np.int32)  # 测试源节点
test_time = test_df['time'].values.astype(np.int32)  # 测试时间戳
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)  # 测试候选目标节点

print(f'Train+Val: {len(df)}, Test: {len(test_df)}')

# 数据分割：15% 用于验证，85% 用于训练
num_total = len(df)
num_val = int(num_total * 0.15)
num_train = num_total - num_val

# 创建训练、验证和完整数据集
train_data = TemporalData(
    src=jt.Var(src_np[:num_train]),
    dst=jt.Var(dst_np[:num_train]),
    t=jt.Var(t_np[:num_train]),
    edge_ids=jt.Var(edge_ids_np[:num_train])
)
val_data = TemporalData(
    src=jt.Var(src_np[num_train:]),
    dst=jt.Var(dst_np[num_train:]),
    t=jt.Var(t_np[num_train:]),
    edge_ids=jt.Var(edge_ids_np[num_train:])
)
full_data = TemporalData(
    src=jt.Var(src_np),
    dst=jt.Var(dst_np),
    t=jt.Var(t_np),
    edge_ids=jt.Var(edge_ids_np)
)

# 创建数据加载器
train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=1.0)
val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=1.0)

# 创建邻居采样器
full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=1)

# 计算节点数量和最小索引
max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
dst_min = min(int(dst_np.min()), int(test_candidates.min()))
src_min = int(src_np.min())

print(f'Node size: {node_size}, Src min: {src_min}, Dst min: {dst_min}')

num_neighbors = 30  # 邻居数量
# 初始化 CRAFT 模型
model = CRAFT(
    n_layers=2, n_heads=2, hidden_size=64, hidden_dropout_prob=0.1, attn_dropout_prob=0.1,
    hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02, n_nodes=node_size,
    max_seq_length=num_neighbors, loss_type='BPR', use_pos=True, input_cat_time_intervals=False,
    output_cat_time_intervals=True, output_cat_repeat_times=True, num_output_layer=1,
    emb_dropout_prob=0.1, skip_connection=True
)
model.set_min_idx(src_min, dst_min)  # 设置最小索引
optimizer = jt.nn.Adam(list(model.parameters()), lr=0.0001)  # Adam 优化器

save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)  # 创建保存目录

# # 如果存在最佳模型文件，则载入作为训练的起点（resume from best）
# best_model_path = f'{save_path}/{args.dataset}_CRAFT_best.pkl'
# if os.path.exists(best_model_path):
#     try:
#         model.load_state_dict(jt.load(best_model_path))
#         print(f'Loaded existing best model {best_model_path}; resuming training from it')
#     except Exception as e:
#         print(f'Warning: failed to load best model {best_model_path}: {e}; training from scratch')
# else:
#     print('No existing best model found; training from scratch')

# 训练模型
print(f'\nTraining for {args.epochs} epoch(s) with early stopping (patience={args.early_stop})...')
best_ap = train(model, optimizer, train_loader, val_loader, full_neighbor_sampler, num_neighbors, args.epochs, save_path, args.dataset, args.early_stop)

# 加载最佳模型进行预测
print('\nGenerating predictions using best model...')
best_model_path = f'{save_path}/{args.dataset}_CRAFT_best.pkl'
if os.path.exists(best_model_path):
    model.load_state_dict(jt.load(best_model_path))
    print(f'Loaded best model from {best_model_path}')
else:
    model.load_state_dict(jt.load(f'{save_path}/{args.dataset}_CRAFT.pkl'))
    print(f'Best model not found, using latest model')

# 生成测试集预测分数
scores = test_competition(model, test_src, test_time, test_candidates, full_neighbor_sampler, num_neighbors, args.batch_size)

print(f'Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]')

# 保存预测结果
output_file = f'{args.output_dir}/{args.dataset}/{args.dataset}_result.csv'
os.makedirs(osp.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    for row in scores:
        f.write(','.join([f'{p:.8f}' for p in row]) + '\n')

print('\n' + '='*80)
print('DONE')
print('='*80)

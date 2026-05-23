"""
更改说明：
1. 增加了负采样参数 default=10
2. 更改了test评价体系，AP->MRR 
3. 新增：构建困难负样本池
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


def test_val_mrr(model, loader, full_neighbor_sampler, num_neighbors):
    """
    使用 MRR@100 作为验证指标，完全对齐比赛要求。
    """
    model.eval()
    mrr_list = []
    
    # 设定验证集的候选节点总数（1个正样本 + 99个负样本 = 100）
    NUM_CANDIDATES = 100 

    loader_tqdm = tqdm(loader, ncols=120, desc='Validation (MRR)')
    for _, batch_data in enumerate(loader_tqdm):
        # 1. 提取基础数据
        src = jt.array(batch_data.src)
        dst = jt.array(batch_data.dst) # 真实的 positive destination
        t = jt.array(batch_data.t)
        
        batch_size = src.shape[0]

        # 2. 为每个正样本动态生成 99 个负样本 (模拟测试集环境)
        # 注意：这里我们使用 np.random.randint 在节点范围内随机采样
        # 更好的做法是从所有可能节点中排除真实的 dst，这里为了效率简化处理
        # 正确代码：上限直接使用 model.n_nodes (因为 randint 的上限是开区间，不会包含 n_nodes 本身)
        neg_dst_np = np.random.randint(model.dst_min_idx, model.n_nodes, size=(batch_size, NUM_CANDIDATES - 1))
        
        # 将正样本和负样本拼接，形状变为 [batch_size, 100]
        # 正样本始终在索引 0 的位置
        pos_item = jt.Var(dst).unsqueeze(1)
        neg_item = jt.Var(neg_dst_np)
        test_dst = jt.cat([pos_item, neg_item], dim=1) 

        # 3. 获取源节点的历史邻居
        src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=src.numpy(), node_interact_times=t.numpy(), num_neighbors=num_neighbors)
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)
        # 4. 获取目标节点(100个)的最后更新时间
        dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=test_dst.flatten().numpy(),
            node_interact_times=np.broadcast_to(t.numpy()[:,np.newaxis], (len(t), test_dst.shape[1])).flatten(),
            num_neighbors=1)
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        # 5. 前向传播计算分数
        # 极其重要：验证集必须使用与 test_competition 完全一致的底层推理逻辑！
        # 这里我们不再使用 model.predict，而是统一使用 model.forward 并手动处理偏移
        
        # 索引偏移处理 (与 test_competition 保持绝对一致)
        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

        # 调用 forward
        logits = model.forward(src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times),
                              jt.Var(t), test_dst=test_dst_adj, dst_last_update_times=dst_last_update_time)
        
        # 获取概率，形状为 [batch_size, 100]
        probs = jt.sigmoid(logits.squeeze(-1)).numpy() 

        # 6. 计算 MRR
        # 由于我们构造 test_dst 时，正样本始终在第 0 列
        # 因此我们需要知道第 0 列的分数在所有 100 个分数中的排名
        for i in range(batch_size):
            scores = probs[i]
            pos_score = scores[0] # 真实节点的预测得分
            
            # 统计有多少个节点的分数严格大于正样本的分数
            # 加 1 是因为如果有 0 个大于它，它的排名就是 1
            rank = np.sum(scores > pos_score) + 1 
            
            mrr_list.append(1.0 / rank)

    return {'MRR': np.mean(mrr_list)}

def train(model, optimizer, train_loader, val_loader, full_neighbor_sampler, num_neighbors, num_epochs, save_path, dataset_name, popular_items, early_stop_patience=10):
    # 注意：函数签名增加了 popular_items 参数
    best_mrr = 0  # 最佳验证 MRR
    patience_counter = 0  # 早停计数器

    for epoch in range(num_epochs):
        model.train()  
        train_losses = []  
        train_tqdm = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch+1}')  

        for batch_idx, batch_data in enumerate(train_tqdm):
            current_neg_ratio = len(batch_data.neg_dst) // len(batch_data.src)

            # 核心魔法：使用 np.repeat 将正样本展开
            src_np = np.repeat(batch_data.src, current_neg_ratio)
            dst_np = np.repeat(batch_data.dst, current_neg_ratio)
            t_np = np.repeat(batch_data.t, current_neg_ratio)

            # ================= [ 新增：动态困难负采样逻辑 ] =================
            original_neg_dst_np = batch_data.neg_dst
            num_total_negs = len(original_neg_dst_np)
            
            # 从热门池中随机抽取与负样本数量相等的“困难节点”
            hard_negs = np.random.choice(popular_items, size=num_total_negs)
            
            # 生成 50% 的替换掩码 (True 代表准备替换为困难样本)
            replace_mask = np.random.rand(num_total_negs) < 0.5
            
            # 【致命错误防御】：确保抽到的困难负样本，不能恰好等于真实的正样本 (dst_np)
            # 如果撞车了，就在该位置取消替换，保留 DataLoader 原本的随机负样本
            conflict_mask = (hard_negs == dst_np)
            final_replace_mask = replace_mask & (~conflict_mask)
            
            # 执行混合替换：掩码为 True 取 hard_negs，False 保留原随机负样本
            mixed_neg_dst_np = np.where(final_replace_mask, hard_negs, original_neg_dst_np)
            # ================================================================

            # 转换为 Jittor 变量
            src = jt.array(src_np)
            dst = jt.array(dst_np)
            t = jt.array(t_np)
            neg_dst = jt.array(mixed_neg_dst_np) # 使用混合后的强力负样本

            # 获取邻居信息
            src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=src.numpy(), node_interact_times=t.numpy(), num_neighbors=num_neighbors)
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)

            if neighbor_num.sum() == 0:  
                continue

            current_batch_size = src.shape[0]
            pos_item = jt.Var(dst).unsqueeze(1) 
            neg_item = jt.Var(neg_dst).reshape(current_batch_size, -1) 
            test_dst = jt.cat([pos_item, neg_item], dim=1)

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

            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()  
            train_losses.append(loss.item())
            train_tqdm.set_description(f'Epoch {epoch+1}, loss: {loss.item():.4f}')

        print(f'Epoch {epoch+1}, Train Loss: {np.mean(train_losses):.4f}')
        
        # 验证
        val_res = test_val_mrr(model, val_loader, full_neighbor_sampler, num_neighbors)
        print(f'Epoch {epoch+1}, Val: {val_res}')

        # 早停检查
        current_mrr = val_res['MRR']
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            jt.save(model.state_dict(), f'{save_path}/{dataset_name}_CRAFT_best.pkl')  
            print(f'  -> New best MRR: {best_mrr:.6f}, model saved!')
        else:
            patience_counter += 1
            print(f'  -> No improvement for {patience_counter} epoch(s), best MRR: {best_mrr:.6f}')

        jt.save(model.state_dict(), f'{save_path}/{dataset_name}_CRAFT.pkl')

        if patience_counter >= early_stop_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs!')
            print(f'Best validation MRR: {best_mrr:.6f}')
            break

    return best_mrr

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
parser.add_argument('--neg_ratio', type=int, default=10, help='训练时的负采样比例')
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

# ================= [ 新增：构建困难负样本池 ] =================
print("正在构建困难负样本池 (Popular Items Pool)...")
# 统计目标节点出现的频次
dst_counts = df['dst'].value_counts()
# 取 Top 1000 个最热门的节点（如果数据集很小，自动取实际数量）
top_k = min(1000, len(dst_counts))
popular_items = dst_counts.head(top_k).index.values.astype(np.int32)
print(f"提取了 {top_k} 个高频热门节点作为困难负样本候选。")
# ==============================================================

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
train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=args.neg_ratio)
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
    n_layers=2, n_heads=4, hidden_size=128, hidden_dropout_prob=0.1, attn_dropout_prob=0.1,
    hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02, n_nodes=node_size,
    max_seq_length=num_neighbors, loss_type='BPR', use_pos=True, input_cat_time_intervals=False,
    output_cat_time_intervals=True, output_cat_repeat_times=True, num_output_layer=1,
    emb_dropout_prob=0.1, skip_connection=True
)
model.set_min_idx(src_min, dst_min)  # 设置最小索引
# 修改 main.py 中的优化器定义
optimizer = jt.nn.Adam(list(model.parameters()), lr=0.0001, weight_decay=1e-4) # Adam 优化器
#optimizer = jt.nn.Adam(list(model.parameters()), lr=0.0001)
save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)  # 创建保存目录

# 训练模型
print(f'\nTraining for {args.epochs} epoch(s) with early stopping (patience={args.early_stop})...')
best_mrr = train(model, optimizer, train_loader, val_loader, full_neighbor_sampler, num_neighbors, args.epochs, save_path, args.dataset, popular_items, args.early_stop)
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

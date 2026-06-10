
"""
更改说明：
1. 增加了负采样参数 default=10
2. 更改了test评价体系，AP->MRR 
3. 新增：gcn增强的混合模型 Hybrid_CRAFT_GCN，注释掉了热门节点候选。添加alpha均衡gcn与craft权重
4. 优化了gcn板块中的时间衰减机制，通过数据集的时间密集程度自动选择最佳分母
5. 【当前版本核心】：双模平行影子测试 (Shadow Testing)！
   同时独立训练【专属 Embedding Alpha】与【全局统一 Alpha】两个模型，
   每个 Epoch 结束后输出 MRR 极差对比 CSV 供 Case Study！
"""
import os
import os.path as osp
import sys
from sympy import true

# 将 JittorGeometric 添加到 Python 路径中，使其可以被导入
root = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, osp.join(root, 'JittorGeometric'))

# 设置 Jittor 同步模式，确保操作同步执行
os.environ['JT_SYNC'] = '1'

# 导入必要的库
import jittor as jt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.data import TemporalData
from jittor_geometric.nn.models.craft import CRAFT
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
import argparse

jt.flags.use_cuda = 1
def test_val_ensemble_mrr(model_emb, model_glo, loader, full_neighbor_sampler, num_neighbors, valid_nodes):
    """
    专门用于在训练过程中，每个 Epoch 测试【软融合】在验证集上的 MRR 成绩
    """
    model_emb.eval()
    model_glo.eval()
    mrr_list = []
    NUM_CANDIDATES = 100 

    loader_tqdm = tqdm(loader, ncols=120, desc='Validation (Soft Ensemble 50/50)', leave=False)
    with jt.no_grad(): # 节省显存
        for _, batch_data in enumerate(loader_tqdm):
            src = jt.array(batch_data.src)
            dst = jt.array(batch_data.dst)
            t = jt.array(batch_data.t)
            batch_size = src.shape[0]

            # 构造验证集的负样本
            neg_dst_np = np.random.choice(valid_nodes, size=(batch_size, NUM_CANDIDATES - 1), replace=True)
            pos_item = jt.Var(dst).unsqueeze(1)
            neg_item = jt.Var(neg_dst_np)
            test_dst = jt.cat([pos_item, neg_item], dim=1) 

            # 获取历史邻居图谱
            max_supported_id = len(full_neighbor_sampler.nodes_neighbor_times) - 1
            src_ids = src.numpy()
            src_oob_mask = src_ids > max_supported_id
            safe_src_ids = np.where(src_oob_mask, 0, src_ids)

            src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=safe_src_ids, node_interact_times=t.numpy(), num_neighbors=num_neighbors)
            src_neighb_seq[src_oob_mask] = 0
            src_neighb_interact_times[src_oob_mask] = 0
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)

            dst_ids = test_dst.flatten().numpy()
            dst_oob_mask = dst_ids > max_supported_id
            safe_dst_ids = np.where(dst_oob_mask, 0, dst_ids)

            dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=safe_dst_ids,
                node_interact_times=np.broadcast_to(t.numpy()[:,np.newaxis], (len(t), test_dst.shape[1])).flatten(),
                num_neighbors=1)
            
            dst_last_neighbor = np.array(dst_last_neighbor)
            dst_last_neighbor[dst_oob_mask] = 0
            dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
            dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000
            dst_last_update_time = jt.Var(dst_last_update_time)

            src_neighb_seq_adj = jt.Var(src_neighb_seq) - model_emb.dst_min_idx + 1
            test_dst_adj = test_dst - model_emb.dst_min_idx + 1
            src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

            # ================= [ 🚀 核心融合逻辑 ] =================
            # 1. 轨道 A (专属)
            logits_emb = model_emb.forward(src, src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times),
                          jt.Var(t), test_dst_adj, dst_last_update_time, 
                          original_src_neighb_seq=jt.Var(src_neighb_seq), original_test_dst=test_dst)
            probs_emb = jt.sigmoid(logits_emb).numpy()

            # 2. 轨道 B (全局)
            logits_glo = model_glo.forward(src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times),
                          jt.Var(t), test_dst_adj, dst_last_update_time, 
                          original_src_neighb_seq=jt.Var(src_neighb_seq), original_test_dst=test_dst)
            probs_glo = jt.sigmoid(logits_glo).numpy()
            
            # 3. 概率按比例相加
            final_probs = 0.5 * probs_emb + 0.5 * probs_glo
            # =======================================================

            # 计算 MRR (第0个是正样本，后面99个是负样本)
            for i in range(batch_size):
                scores = final_probs[i]
                pos_score = scores[0] 
                rank = np.sum(scores > pos_score) + 1 
                mrr_list.append(1.0 / rank)

    return {'MRR': np.mean(mrr_list)}
def test_val_mrr(model, loader, full_neighbor_sampler, num_neighbors, valid_nodes, is_emb_model=False):
    """
    使用 MRR@100 作为验证指标，完全对齐比赛要求。
    增加了 is_emb_model 参数以适配两种模型的 forward 签名
    """
    model.eval()
    mrr_list = []
    NUM_CANDIDATES = 100 

    loader_tqdm = tqdm(loader, ncols=120, desc='Validation (MRR)', leave=False)
    for _, batch_data in enumerate(loader_tqdm):
        src = jt.array(batch_data.src)
        dst = jt.array(batch_data.dst)
        t = jt.array(batch_data.t)
        batch_size = src.shape[0]

        neg_dst_np = np.random.choice(valid_nodes, size=(batch_size, NUM_CANDIDATES - 1), replace=True)
        pos_item = jt.Var(dst).unsqueeze(1)
        neg_item = jt.Var(neg_dst_np)
        test_dst = jt.cat([pos_item, neg_item], dim=1) 

        max_supported_id = len(full_neighbor_sampler.nodes_neighbor_times) - 1
        src_ids = src.numpy()
        src_oob_mask = src_ids > max_supported_id
        safe_src_ids = np.where(src_oob_mask, 0, src_ids)

        src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=safe_src_ids, node_interact_times=t.numpy(), num_neighbors=num_neighbors)
        src_neighb_seq[src_oob_mask] = 0
        src_neighb_interact_times[src_oob_mask] = 0
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        dst_ids = test_dst.flatten().numpy()
        dst_oob_mask = dst_ids > max_supported_id
        safe_dst_ids = np.where(dst_oob_mask, 0, dst_ids)

        dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=safe_dst_ids,
            node_interact_times=np.broadcast_to(t.numpy()[:,np.newaxis], (len(t), test_dst.shape[1])).flatten(),
            num_neighbors=1)
        
        dst_last_neighbor = np.array(dst_last_neighbor)
        dst_last_neighbor[dst_oob_mask] = 0
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

        if is_emb_model:
            logits = model.forward(src, src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times),
                          jt.Var(t), test_dst_adj, dst_last_update_time, 
                          original_src_neighb_seq=jt.Var(src_neighb_seq), original_test_dst=test_dst)
        else:
            logits = model.forward(src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times),
                          jt.Var(t), test_dst_adj, dst_last_update_time, 
                          original_src_neighb_seq=jt.Var(src_neighb_seq), original_test_dst=test_dst)
            
        probs = jt.sigmoid(logits).numpy()

        for i in range(batch_size):
            scores = probs[i]
            pos_score = scores[0] 
            rank = np.sum(scores > pos_score) + 1 
            mrr_list.append(1.0 / rank)

    return {'MRR': np.mean(mrr_list)}

# ==============================================================================
# 🚀 独立模型 A：专属身份证版本 (Hybrid_CRAFT_GCN_Emb)
# ==============================================================================
class Hybrid_CRAFT_GCN_Emb(jt.nn.Module):
    def __init__(self, craft_model, n_nodes, hidden_dim=128, time_scale=100000.0):
        super().__init__()
        self.craft = craft_model
        self.n_nodes = n_nodes 
        self.gcn_emb = jt.nn.Embedding(n_nodes, hidden_dim)
        self.gcn_emb.weight = jt.randn(self.gcn_emb.weight.shape) * 0.01
        
        # 专属身份证 Alpha 参数
        self.user_alpha_emb = jt.nn.Embedding(n_nodes, 1)
        self.user_alpha_emb.weight = jt.zeros(self.user_alpha_emb.weight.shape)
        
        self.gcn_scale = jt.ones((1,)) * 20.0 
        self.time_scale = time_scale

    @property
    def dst_min_idx(self): return self.craft.dst_min_idx
    @property
    def src_min_idx(self): return self.craft.src_min_idx

    def get_gcn_logits(self, src_neighb_seq, neighbor_num, test_dst, src_neighb_interact_times, cur_pred_times, print_debug=False):
        neighb_embs = self.gcn_emb(src_neighb_seq) 
        mask = (src_neighb_seq > 0).float()
        sum_neighb_embs = (neighb_embs * mask.unsqueeze(-1)).sum(dim=1)
        src_gcn_emb = sum_neighb_embs / (neighbor_num.unsqueeze(1).float() + 1e-8)
        
        dst_embs = self.gcn_emb(test_dst)
        src_gcn_norm = src_gcn_emb / (jt.norm(src_gcn_emb, dim=-1, keepdims=True) + 1e-8)
        dst_embs_norm = dst_embs / (jt.norm(dst_embs, dim=-1, keepdims=True) + 1e-8)
        
        gcn_logits = jt.sum(src_gcn_norm.unsqueeze(1) * dst_embs_norm, dim=-1)
        gcn_logits = gcn_logits * self.gcn_scale
        return gcn_logits, src_gcn_emb

    def calculate_loss(self, src_ids, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times, print_debug=False):
        src_neighb_seq_adj = jt.Var(src_neighb_seq) - self.craft.dst_min_idx + 1
        test_dst_adj = test_dst - self.craft.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)
        
        craft_logits = self.craft.forward(src_neighb_seq_adj, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst_adj, dst_last_update_times).squeeze(-1)
        gcn_logits, _ = self.get_gcn_logits(src_neighb_seq, src_neighb_seq_len, test_dst, src_neighb_interact_times, cur_pred_times, print_debug=print_debug)
        
        alpha_emb = jt.sigmoid(self.user_alpha_emb(src_ids)).squeeze(-1)
        final_logits_emb = (1.0 - alpha_emb.unsqueeze(1)) * craft_logits + alpha_emb.unsqueeze(1) * gcn_logits 
        
        # 带有动态学习率杠杆的 BPR
        pos_score_emb = final_logits_emb[:, 0]
        neg_score_emb = final_logits_emb[:, 1:]
        pos_score_expanded_emb = pos_score_emb.unsqueeze(1).repeat(1, neg_score_emb.shape[1])
        bpr_loss_matrix = -jt.log(jt.sigmoid(pos_score_expanded_emb - neg_score_emb) + 1e-8)
        user_loss = bpr_loss_matrix.mean(dim=1) 
        
        loss = user_loss.mean()
        
        return loss, pos_score_emb, neg_score_emb, alpha_emb.mean()

    def forward(self, src_ids, src_neighb_seq_adj, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst_adj, dst_last_update_times, original_src_neighb_seq, original_test_dst):
        craft_logits = self.craft.forward(src_neighb_seq_adj, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst_adj, dst_last_update_times).squeeze(-1)
        gcn_logits, _ = self.get_gcn_logits(original_src_neighb_seq, src_neighb_seq_len, original_test_dst, src_neighb_interact_times, cur_pred_times)
        
        alpha_emb = jt.sigmoid(self.user_alpha_emb(src_ids)).squeeze(-1)
        final_logits_emb = (1.0 - alpha_emb.unsqueeze(1)) * craft_logits + alpha_emb.unsqueeze(1) * gcn_logits
        return final_logits_emb

# ==============================================================================
# 🚀 独立模型 B：全局吃大锅饭版本 (Hybrid_CRAFT_GCN_Global)
# ==============================================================================
class Hybrid_CRAFT_GCN_Global(jt.nn.Module):
    def __init__(self, craft_model, n_nodes, hidden_dim=128, time_scale=100000.0):
        super().__init__()
        self.craft = craft_model
        self.n_nodes = n_nodes 
        self.gcn_emb = jt.nn.Embedding(n_nodes, hidden_dim)
        self.gcn_emb.weight = jt.randn(self.gcn_emb.weight.shape) * 0.01
        
        # 全局统一 Alpha 参数
        self.fusion_weight = jt.zeros((1,))
        
        self.gcn_scale = jt.ones((1,)) * 20.0 
        self.time_scale = time_scale

    @property
    def dst_min_idx(self): return self.craft.dst_min_idx
    @property
    def src_min_idx(self): return self.craft.src_min_idx

    def get_gcn_logits(self, src_neighb_seq, neighbor_num, test_dst, src_neighb_interact_times, cur_pred_times, print_debug=False):
        neighb_embs = self.gcn_emb(src_neighb_seq) 
        mask = (src_neighb_seq > 0).float()
        sum_neighb_embs = (neighb_embs * mask.unsqueeze(-1)).sum(dim=1)
        src_gcn_emb = sum_neighb_embs / (neighbor_num.unsqueeze(1).float() + 1e-8)
        
        dst_embs = self.gcn_emb(test_dst)
        src_gcn_norm = src_gcn_emb / (jt.norm(src_gcn_emb, dim=-1, keepdims=True) + 1e-8)
        dst_embs_norm = dst_embs / (jt.norm(dst_embs, dim=-1, keepdims=True) + 1e-8)
        
        gcn_logits = jt.sum(src_gcn_norm.unsqueeze(1) * dst_embs_norm, dim=-1)
        gcn_logits = gcn_logits * self.gcn_scale
        return gcn_logits

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times, print_debug=False):
        src_neighb_seq_adj = jt.Var(src_neighb_seq) - self.craft.dst_min_idx + 1
        test_dst_adj = test_dst - self.craft.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)
        
        craft_logits = self.craft.forward(src_neighb_seq_adj, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst_adj, dst_last_update_times).squeeze(-1)
        gcn_logits = self.get_gcn_logits(src_neighb_seq, src_neighb_seq_len, test_dst, src_neighb_interact_times, cur_pred_times, print_debug=print_debug)
        
        alpha_global = jt.sigmoid(self.fusion_weight)
        final_logits_global = (1.0 - alpha_global) * craft_logits + alpha_global * gcn_logits
        
        # 标准 BPR Loss
        pos_score = final_logits_global[:, 0]
        neg_score = final_logits_global[:, 1:]
        pos_score_expanded = pos_score.unsqueeze(1).repeat(1, neg_score.shape[1])
        loss = -jt.log(jt.sigmoid(pos_score_expanded - neg_score) + 1e-8).mean()
        
        return loss, pos_score, neg_score, alpha_global

    def forward(self, src_neighb_seq_adj, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst_adj, dst_last_update_times, original_src_neighb_seq, original_test_dst):
        craft_logits = self.craft.forward(src_neighb_seq_adj, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst_adj, dst_last_update_times).squeeze(-1)
        gcn_logits = self.get_gcn_logits(original_src_neighb_seq, src_neighb_seq_len, original_test_dst, src_neighb_interact_times, cur_pred_times)
        
        alpha_global = jt.sigmoid(self.fusion_weight)
        final_logits = (1.0 - alpha_global) * craft_logits + alpha_global * gcn_logits
        return final_logits

# ==============================================================================
# 📊 双开分析实验室：每个 Epoch 跑完揪出极差样本
# ==============================================================================
def analyze_dual_models(model_emb, model_global, loader, full_neighbor_sampler, num_neighbors, valid_nodes, save_dir, dataset_name, epoch_idx):
    print("\n" + "🔍"*20)
    print(f"正在执行 Epoch {epoch_idx} 双模影子测试 (Shadow Test) 解剖...")
    model_emb.eval()
    model_global.eval()
    analysis_records = []
    NUM_CANDIDATES = 100 
    max_supported_id = len(full_neighbor_sampler.nodes_neighbor_times) - 1

    loader_tqdm = tqdm(loader, ncols=120, desc=f'Analyzing Gap E{epoch_idx}')
    for _, batch_data in enumerate(loader_tqdm):
        src = jt.array(batch_data.src)
        dst = jt.array(batch_data.dst)
        t = jt.array(batch_data.t)
        batch_size = src.shape[0]

        neg_dst_np = np.random.choice(valid_nodes, size=(batch_size, NUM_CANDIDATES - 1), replace=True)
        pos_item = jt.Var(dst).unsqueeze(1)
        neg_item = jt.Var(neg_dst_np)
        test_dst = jt.cat([pos_item, neg_item], dim=1) 

        src_ids = src.numpy()
        src_oob_mask = src_ids > max_supported_id
        safe_src_ids = np.where(src_oob_mask, 0, src_ids)
        src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=safe_src_ids, node_interact_times=t.numpy(), num_neighbors=num_neighbors)
        src_neighb_seq[src_oob_mask] = 0
        src_neighb_interact_times[src_oob_mask] = 0
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        dst_ids = test_dst.flatten().numpy()
        dst_oob_mask = dst_ids > max_supported_id
        safe_dst_ids = np.where(dst_oob_mask, 0, dst_ids)
        dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=safe_dst_ids, node_interact_times=np.broadcast_to(t.numpy()[:,np.newaxis], (len(t), test_dst.shape[1])).flatten(), num_neighbors=1)
        dst_last_neighbor = np.array(dst_last_neighbor)
        dst_last_neighbor[dst_oob_mask] = 0
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model_emb.dst_min_idx + 1
        test_dst_adj = test_dst - model_emb.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

        # 🏃 轨道 A：专属 Emb 推理
        logits_emb = model_emb.forward(src, src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times), jt.Var(t), test_dst_adj, dst_last_update_time, jt.Var(src_neighb_seq), test_dst)
        probs_emb = jt.sigmoid(logits_emb).numpy()
        mrr_emb = 1.0 / (np.sum(probs_emb > probs_emb[:, 0:1], axis=1) + 1)
        alpha_emb = jt.sigmoid(model_emb.user_alpha_emb(src)).squeeze(-1).numpy()

        # 🏃 轨道 B：全局 Global 推理
        logits_glo = model_global.forward(src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times), jt.Var(t), test_dst_adj, dst_last_update_time, jt.Var(src_neighb_seq), test_dst)
        probs_glo = jt.sigmoid(logits_glo).numpy()
        mrr_glo = 1.0 / (np.sum(probs_glo > probs_glo[:, 0:1], axis=1) + 1)
        alpha_glo_val = jt.sigmoid(model_global.fusion_weight).item()

        src_np = src.numpy()
        for i in range(batch_size):
            delta_mrr = mrr_emb[i] - mrr_glo[i]
            # 记录差异大于 0.05 的极差样本
            if abs(delta_mrr) > 0.05:
                analysis_records.append({
                    'Epoch': epoch_idx,
                    'src_id': src_np[i],
                    'Alpha_Emb(专属)': round(alpha_emb[i], 4),
                    'Alpha_Glo(全局)': round(alpha_glo_val, 4),
                    'MRR_Emb(专属)': round(mrr_emb[i], 4),
                    'MRR_Glo(全局)': round(mrr_glo[i], 4),
                    'Delta_MRR (Emb - Glo)': round(delta_mrr, 4)
                })

    if len(analysis_records) > 0:
        df_analysis = pd.DataFrame(analysis_records)
        df_analysis['Abs_Delta'] = df_analysis['Delta_MRR (Emb - Glo)'].abs()
        df_analysis = df_analysis.sort_values(by='Abs_Delta', ascending=False).drop(columns=['Abs_Delta'])
        
        out_path = f"{save_dir}/{dataset_name}_Epoch{epoch_idx}_ShadowTest_Analysis.csv"
        df_analysis.to_csv(out_path, index=False)
        print(f"\n[生成解剖报告] 提取了 {len(df_analysis)} 条极差样本 -> {out_path}\n")
    else:
        print("\n[平局] 两个平行模型本轮表现完全一致。\n")


# ==============================================================================
# 🎮 双开训练主循环
# ==============================================================================
def train(model_emb, model_global, opt_emb, opt_global, train_loader, val_loader, full_neighbor_sampler, num_neighbors, num_epochs, save_path, dataset_name, valid_nodes, early_stop_patience=10):
    best_mrr_emb = 0  
    best_mrr_glo = 0
    patience_counter = 0  

    for epoch in range(num_epochs):
        model_emb.train()  
        model_global.train()
        train_tqdm = tqdm(train_loader, ncols=140, desc=f'Epoch {epoch+1}',mininterval=0.5) 

        for batch_idx, batch_data in enumerate(train_tqdm):
            raw_batch_size = len(batch_data.src) 
            
            # 2. 计算每一笔正样本，分到了几个负样本名额（比如 10 个）
            current_neg_ratio = len(batch_data.neg_dst) // raw_batch_size
            
            # =====================================================================
            # 👑 核心修复：混合负采样逻辑 (1D 展平对齐)
            # =====================================================================
            num_hard = int(current_neg_ratio * args.hard_neg_ratio)
            num_easy = current_neg_ratio - num_hard
            
            # 抽取的形状必须是 (raw_batch_size, num_xxx)，确保每行各司其职
            if num_hard > 0:
                hard_negs = np.random.choice(popular_items, size=(raw_batch_size, num_hard), replace=True)
            else:
                hard_negs = np.empty((raw_batch_size, 0), dtype=np.int32)
                
            if num_easy > 0:
                easy_negs = np.random.choice(valid_nodes_pool, size=(raw_batch_size, num_easy), replace=True)
            else:
                easy_negs = np.empty((raw_batch_size, 0), dtype=np.int32)
                
            # 核心细节：横向拼接得到 (200, 10)，然后【逐行按顺序展平】
            # 这样展平后的顺序就是：[User1的10个负样本, User2的10个负样本, ...]
            combined_negs = np.concatenate([hard_negs, easy_negs], axis=1)
            neg_dst_np = combined_negs.flatten() # 长度 2000
            # =====================================================================

            # 3. 顺应原代码的“膨胀模式”，把正样本和时间戳重复 N 倍展平
            src_np = np.repeat(batch_data.src, current_neg_ratio) # 长度 2000
            dst_np = np.repeat(batch_data.dst, current_neg_ratio) # 长度 2000
            t_np = np.repeat(batch_data.t, current_neg_ratio)     # 长度 2000

            src = jt.array(src_np)
            dst = jt.array(dst_np)
            t = jt.array(t_np)
            neg_dst = jt.array(neg_dst_np) # 长度 2000

            src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=src.numpy(), node_interact_times=t.numpy(), num_neighbors=num_neighbors)
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)

            if neighbor_num.sum() == 0:  
                continue
            current_batch_size = src.shape[0] 
            
            pos_item = jt.Var(dst).unsqueeze(1) 
            neg_item = jt.Var(neg_dst).unsqueeze(1)
            
            test_dst = jt.cat([pos_item, neg_item], dim=1)

            # =====================================================================
            # 🛡️ 终极防御层：严格限幅，干掉任何导致 IndexError 的越界恶魔 ID
            # =====================================================================
            # 1. 探测采样器能支持的最大合法索引上限
            max_supported_id = len(full_neighbor_sampler.nodes_neighbor_times) - 1
            
            # 2. 先转成干净的 1D NumPy 整数数组
            dst_ids_np = test_dst.flatten().int().numpy()
            
            # 3. 拦截任何大于上限或小于 0 的非法节点，统统重置为安全的 0 节点
            oob_mask = (dst_ids_np > max_supported_id) | (dst_ids_np < 0)
            safe_dst_ids_np = np.where(oob_mask, 0, dst_ids_np)
            # =====================================================================

            # 把防越界的安全 ID 喂给采样器，绝对不可能再报 IndexError！
            dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=safe_dst_ids_np,
                node_interact_times=np.broadcast_to(t.numpy()[:,np.newaxis], (len(t), test_dst.shape[1])).flatten(),
                num_neighbors=1)
                
            dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
            dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000
            dst_last_update_time = jt.Var(dst_last_update_time)

            src_neighb_seq_adj = jt.Var(src_neighb_seq) - model_emb.dst_min_idx + 1
            
            # 🌟 同样的，送入模型前也做一个安全的类型和越界控制
            test_dst_adj = jt.Var(safe_dst_ids_np).reshape(test_dst.shape) - model_emb.dst_min_idx + 1
            test_dst_adj = jt.where(test_dst_adj < 0, jt.zeros_like(test_dst_adj), test_dst_adj)
            
            src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

            # ================= [ 🚀 赛道 A：专属 Emb 模型训练 ] =================
            loss_emb, _, _, alpha_emb = model_emb.calculate_loss(
                src_ids=src, src_neighb_seq=jt.Var(src_neighb_seq), src_neighb_seq_len=jt.Var(neighbor_num),
                src_neighb_interact_times=jt.Var(src_neighb_interact_times), cur_pred_times=jt.Var(t),
                test_dst=test_dst, dst_last_update_times=dst_last_update_time)
            
            opt_emb.zero_grad()
            opt_emb.step(loss_emb)

            # ================= [ 🚀 赛道 B：全局 Global 模型训练 ] =================
            loss_glo, _, _, alpha_glo = model_global.calculate_loss(
                src_neighb_seq=jt.Var(src_neighb_seq), src_neighb_seq_len=jt.Var(neighbor_num),
                src_neighb_interact_times=jt.Var(src_neighb_interact_times), cur_pred_times=jt.Var(t),
                test_dst=test_dst, dst_last_update_times=dst_last_update_time)
            
            opt_global.zero_grad()
            opt_global.step(loss_glo)

            jt.sync_all()  
            train_tqdm.set_description(f'E{epoch+1} | Emb_L:{loss_emb.item():.2f} a:{alpha_emb.item():.2f} | Glo_L:{loss_glo.item():.2f} a:{alpha_glo.item():.2f}')

        # 🏃 验证模型 A
        val_res_emb = test_val_mrr(model_emb, val_loader, full_neighbor_sampler, num_neighbors, valid_nodes, is_emb_model=True)
        print(f'Epoch {epoch+1}, Val Emb (专属): {val_res_emb}')
        # 🏃 验证模型 B
        val_res_glo = test_val_mrr(model_global, val_loader, full_neighbor_sampler, num_neighbors, valid_nodes, is_emb_model=False)
        print(f'Epoch {epoch+1}, Val Glo (全局): {val_res_glo}')
        val_res_ens = test_val_ensemble_mrr(model_emb, model_global, val_loader, full_neighbor_sampler, num_neighbors, valid_nodes)
        print(f'Epoch {epoch+1}, Val Ensemble (软融合): {val_res_ens} 🚀🚀🚀')
        # 🏆 以模型 A (专属 Emb) 作为早停的主指标保存
        current_mrr_emb = val_res_emb['MRR']
        if current_mrr_emb > best_mrr_emb:
            best_mrr_emb = current_mrr_emb
            patience_counter = 0
            jt.save(model_emb.state_dict(), f'{save_path}/{dataset_name}_CRAFT_Emb_best.pkl')  
            jt.save(model_global.state_dict(), f'{save_path}/{dataset_name}_CRAFT_Glo_best.pkl') 
            print(f'  -> New best Emb MRR: {best_mrr_emb:.6f}, both models saved!')
        else:
            patience_counter += 1
            print(f'  -> No improvement for Emb model for {patience_counter} epoch(s), best MRR: {best_mrr_emb:.6f}')

        jt.save(model_emb.state_dict(), f'{save_path}/{dataset_name}_CRAFT_Emb.pkl')
        jt.save(model_global.state_dict(), f'{save_path}/{dataset_name}_CRAFT_Glo.pkl')

        if patience_counter >= early_stop_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs!')
            break

    return best_mrr_emb

def test_competition(model, test_src, test_time, test_candidates, full_neighbor_sampler, num_neighbors, batch_size=200):
    model.eval()
    all_scores = []
    num_samples = len(test_src)
    num_batches = (num_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches), ncols=120, desc='Testing (Using Emb Model)')
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
        safe_dst_ids_np = test_dst.flatten().int().numpy()
        dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=safe_dst_ids_np,
            node_interact_times=np.broadcast_to(batch_time[:,np.newaxis], (len(batch_time), test_dst.shape[1])).flatten(),
            num_neighbors=1)
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

        logits = model.forward(jt.Var(batch_src), src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times),
                              jt.Var(batch_time), test_dst_adj=test_dst_adj, dst_last_update_times=dst_last_update_time,
                              original_src_neighb_seq=jt.Var(src_neighb_seq), original_test_dst=test_dst)
        probs = jt.sigmoid(logits).numpy()
        all_scores.append(probs)

    return np.vstack(all_scores)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='模型保存目录')
parser.add_argument('--output_dir', type=str, default=None, help='输出目录 (默认与数据目录相同)')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=200, help='批次大小')
parser.add_argument('--neg_ratio', type=int, default=10, help='训练时的负采样比例')
parser.add_argument('--early_stop', type=int, default=10, help='早停耐心值')
parser.add_argument('--resume', action='store_true', help='加上此参数，即可从最新的 checkpoint 恢复训练')
parser.add_argument('--formal',action='store_true',help='加上此参数使用全部训练集')
parser.add_argument('--hard_neg_ratio', type=float, default=0.3, 
                    help='困难负样本的占比 (例如 0.2 表示 20% 的负样本来自热门商品，80% 随机)')
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

print('='*80)
print(f'CRAFT Competition - Dataset: {args.dataset}')
print('='*80)

df = pd.read_csv(f'{args.data_dir}/{args.dataset}/train.csv')
test_df = pd.read_csv(f'{args.data_dir}/{args.dataset}/test.csv')

src_np = df['src'].values.astype(np.int32) 
dst_np = df['dst'].values.astype(np.int32)
t_np = df['time'].values.astype(np.int32) 
edge_ids_np = np.arange(len(df), dtype=np.int32) + 1 
valid_nodes_pool = np.unique(np.concatenate([src_np, dst_np]))
# ================= [ 新增：构建困难负样本池 ] =================
print("\n" + "="*50)
print("🎯 正在构建困难负样本池 (Popular Items Pool)...")
# 统计目标节点出现的频次
dst_counts = df['dst'].value_counts()
# 取 Top 1000 个最热门的节点（如果数据集很小，自动取实际数量）
top_k = min(1000, len(dst_counts))
popular_items = dst_counts.head(top_k).index.values.astype(np.int32)
print(f"提取了 {top_k} 个高频热门节点作为困难负样本候选。")
print(f"当前困难负样本占比设定为: {args.hard_neg_ratio * 100}%")
print("="*50 + "\n")
# ==============================================================
global_time_scale = float(np.std(t_np))
if global_time_scale < 1.0: global_time_scale = 1.0
print(f"🌟 [Data Driven] 嗅探到当前数据集自适应时间缩放基准 (Time Scale): {global_time_scale:.2f}")

test_src = test_df['src'].values.astype(np.int32)
test_time = test_df['time'].values.astype(np.int32)
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

print(f'Train+Val: {len(df)}, Test: {len(test_df)}')
if args.formal:
    train_ratio = 0.0001
else:
    train_ratio = 0.15

num_total = len(df)
num_val = int(num_total * train_ratio)
num_train = num_total - num_val

train_data = TemporalData(src=jt.Var(src_np[:num_train]), dst=jt.Var(dst_np[:num_train]), t=jt.Var(t_np[:num_train]), edge_ids=jt.Var(edge_ids_np[:num_train]))
val_data = TemporalData(src=jt.Var(src_np[num_train:]), dst=jt.Var(dst_np[num_train:]), t=jt.Var(t_np[num_train:]), edge_ids=jt.Var(edge_ids_np[num_train:]))
full_data = TemporalData(src=jt.Var(src_np), dst=jt.Var(dst_np), t=jt.Var(t_np), edge_ids=jt.Var(edge_ids_np))

train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=args.neg_ratio)
val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=1.0)

full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=1)
train_neighbor_sampler = get_neighbor_sampler(train_data, 'recent', seed=1)

max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
dst_min = min(int(dst_np.min()), int(test_candidates.min()))
src_min = int(src_np.min())
print(f'Node size: {node_size}, Src min: {src_min}, Dst min: {dst_min}')
num_neighbors = 30

# ==============================================================================
# 🚀 实例化双开引擎 (CRAFT_Emb vs CRAFT_Glo)
# ==============================================================================
craft_model_emb = CRAFT(
    n_layers=2, n_heads=4, hidden_size=128, hidden_dropout_prob=0.1, attn_dropout_prob=0.1,
    hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02, n_nodes=node_size,
    max_seq_length=num_neighbors, loss_type='BPR', use_pos=True, input_cat_time_intervals=False,
    output_cat_time_intervals=True, output_cat_repeat_times=True, num_output_layer=1, emb_dropout_prob=0.1, skip_connection=True
)
craft_model_emb.set_min_idx(src_min, dst_min)
model_emb = Hybrid_CRAFT_GCN_Emb(craft_model=craft_model_emb, n_nodes=node_size, hidden_dim=128, time_scale=global_time_scale)

# 为了绝对物理隔离，再给全局版实例化一个干净的底座
craft_model_glo = CRAFT(
    n_layers=2, n_heads=4, hidden_size=128, hidden_dropout_prob=0.1, attn_dropout_prob=0.1,
    hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02, n_nodes=node_size,
    max_seq_length=num_neighbors, loss_type='BPR', use_pos=True, input_cat_time_intervals=False,
    output_cat_time_intervals=True, output_cat_repeat_times=True, num_output_layer=1, emb_dropout_prob=0.1, skip_connection=True
)
craft_model_glo.set_min_idx(src_min, dst_min)
model_glo = Hybrid_CRAFT_GCN_Global(craft_model=craft_model_glo, n_nodes=node_size, hidden_dim=128, time_scale=global_time_scale)

# 优化器 A：专属免税打鸡血
alpha_params = model_emb.user_alpha_emb.parameters()
base_params = [p for n, p in model_emb.named_parameters() if 'user_alpha_emb' not in n]
opt_emb = jt.nn.Adam([
    {'params': base_params, 'weight_decay': 1e-4},
    {'params': alpha_params, 'weight_decay': 0.0, 'lr': 0.001}
], lr=0.0001)

# 优化器 B：全局标准版
opt_glo = jt.nn.Adam(list(model_glo.parameters()), lr=0.0001, weight_decay=1e-4)

save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)

latest_emb_path = f'{save_path}/{args.dataset}_CRAFT_Emb.pkl'
latest_glo_path = f'{save_path}/{args.dataset}_CRAFT_Glo.pkl'
if args.resume:
    if os.path.exists(latest_emb_path) and os.path.exists(latest_glo_path):
        model_emb.load_state_dict(jt.load(latest_emb_path))
        model_glo.load_state_dict(jt.load(latest_glo_path))
        print(f'\n[🚀 Resume] 成功加载双引擎权重！')
    else:
        print(f'\n[!] 未找到历史双开文件，将从头开始新一轮训练。')

print(f'\nTraining for {args.epochs} epoch(s) with early stopping (patience={args.early_stop})...')
best_mrr = train(model_emb, model_glo, opt_emb, opt_glo, train_loader, val_loader, train_neighbor_sampler, num_neighbors, args.epochs, save_path, args.dataset, valid_nodes_pool, args.early_stop)

print('\nGenerating predictions using Best Emb model...')
best_emb_path = f'{save_path}/{args.dataset}_CRAFT_Emb_best.pkl'
if os.path.exists(best_emb_path):
    model_emb.load_state_dict(jt.load(best_emb_path))
else:
    model_emb.load_state_dict(jt.load(latest_emb_path))

scores = test_competition(model_emb, test_src, test_time, test_candidates, full_neighbor_sampler, num_neighbors, args.batch_size)

output_file = f'{args.output_dir}/{args.dataset}/{args.dataset}_result.csv'
os.makedirs(osp.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    for row in scores:
        f.write(','.join([f'{p:.8f}' for p in row]) + '\n')

print('\n' + '='*80)
print('DONE')
print('='*80)


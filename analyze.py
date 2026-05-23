import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import argparse

def analyze_temporal_graph(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. 基础时序与规模统计
    num_edges = len(df)
    unique_src = df['src'].nunique()
    unique_dst = df['dst'].nunique()
    total_nodes = len(set(df['src']).union(set(df['dst'])))
    
    print("\n" + "="*40)
    print(" 1. 基础规模特征 (Basic Stats)")
    print("="*40)
    print(f"总交互次数 (Edges): {num_edges}")
    print(f"源节点数 (Users/Src): {unique_src}")
    print(f"目标节点数 (Items/Dst): {unique_dst}")
    print(f"总节点数 (Total Nodes): {total_nodes}")
    
    # 2. 交互重复率 (Sequence vs Graph 的核心指标)
    # 计算 (src, dst) 组合出现的次数
    interaction_counts = df.groupby(['src', 'dst']).size()
    repetitive_ratio = (num_edges - len(interaction_counts)) / num_edges
    
    print("\n" + "="*40)
    print(" 2. 时序动态特征 (Temporal Dynamics)")
    print("="*40)
    print(f"重复交互率 (Repetitive Ratio): {repetitive_ratio:.4f}")
    if repetitive_ratio > 0.5:
        print("  -> [诊断] 高重复率！说明用户倾向于反复与同一物品交互。纯序列模型(RNN/Attention)通常表现很好。")
    else:
        print("  -> [诊断] 低重复率！用户一直在探索新物品。极其需要 GCN 引入协同过滤(朋友买过什么)来辅助预测！")

    # 3. 拓扑结构特征 (使用 NetworkX 构建静态图)
    print("\n" + "="*40)
    print(" 3. 静态图拓扑特征 (Topological Features)")
    print("="*40)
    
    # 构建无向图（忽略时间，仅看拓扑骨架）
    G = nx.from_pandas_edgelist(df, source='src', target='dst', create_using=nx.Graph())
    
    # 连通分量 (看图是不是碎片的)
    connected_components = nx.number_connected_components(G)
    print(f"连通子图数量: {connected_components}")
    
    # 节点度数分布 (找超级节点)
    degrees = [d for n, d in G.degree()]
    max_degree = np.max(degrees)
    mean_degree = np.mean(degrees)
    print(f"平均节点度数 (Mean Degree): {mean_degree:.2f}")
    print(f"最大节点度数 (Max Degree): {max_degree}")
    
    if max_degree > mean_degree * 100:
        print("  -> [诊断] 存在极其活跃的'超级节点'(Hubs)！GCN 在聚合时容易发生过度平滑(Over-smoothing)，需谨慎或使用 Degree-Normalization。")
        
    # 聚集系数 (社区效应)
    # 注意：如果图太大会很慢，可以采样计算
    print("正在计算聚集系数 (可能需要几秒到几分钟)...")
    try:
        # 如果是纯粹的二分图（如纯User-Item，用户间无连边），这个值趋近于0
        clustering_coeff = nx.average_clustering(G)
        print(f"平均聚集系数 (Clustering Coefficient): {clustering_coeff:.4f}")
        
        if clustering_coeff < 0.01:
            print("  -> [诊断] 聚集系数极低。这通常是一个二分图(Bipartite Graph)。普通 GCN 直接用效果差，建议使用 LightGCN 等二分图专门变体。")
        elif clustering_coeff > 0.1:
            print("  -> [诊断] 聚集系数较高！图中存在明显的'圈子'或'社区'。强烈建议引入 GCN 来捕获这种高阶结构信息！")
    except Exception as e:
        print("图太大，聚集系数计算被跳过。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='数据集的train.csv路径')
    args = parser.parse_args()
    analyze_temporal_graph(args.dataset)
import pandas as pd
import numpy as np

# === 修改为你自己的路径 ===
data_dir = './data'
dataset = 'dataset1'   # 例如 'JODIE' 或 'lastfm'
# ===========================

# 读取数据
train_df = pd.read_csv(f'{data_dir}/{dataset}/train.csv')
test_df  = pd.read_csv(f'{data_dir}/{dataset}/test.csv')

# 1. 提取测试集所有候选节点（从第三列开始）
candidate_cols = test_df.columns[2:]   # 'c1' ~ 'c100'
all_candidates = test_df[candidate_cols].values.astype(np.int32).flatten()

# 统计每个候选节点的出现频率
test_freq = pd.Series(all_candidates).value_counts().reset_index()
test_freq.columns = ['node_id', 'freq_in_test']

# 2. 统计训练集中各节点作为 dst 的热度
dst_popularity = train_df['dst'].value_counts()

# 3. 将热度映射到测试候选节点上
test_freq['popularity_in_train'] = test_freq['node_id'].map(dst_popularity).fillna(0).astype(int)

# 4. 保存结果
output_path = f'{data_dir}/{dataset}/candidate_popularity_analysis.csv'
test_freq.to_csv(output_path, index=False)

print(f"结果已保存至: {output_path}")
print("前10条预览：")
print(test_freq.head(10))
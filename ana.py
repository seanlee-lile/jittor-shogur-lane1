import pandas as pd

# ================= 1. 配置文件路径 =================
# 请替换成你实际的文件路径
TRAIN_CSV_PATH = './data/dataset2/train.csv'
AB_TEST_CSV_PATH = './saved_models/dataset2_Epoch1_ShadowTest_Analysis.csv' 

# ================= 2. 读取数据 =================
print("正在加载数据...")
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_ab = pd.read_csv(AB_TEST_CSV_PATH)

# ================= 3. 统计历史交互次数 =================
src_counts = df_train['src'].value_counts().reset_index()
src_counts.columns = ['src_id', '历史出现次数']

# ================= 4. 逐行合并 =================
df_merged = pd.merge(df_ab, src_counts, on='src_id', how='left')
df_merged['历史出现次数'] = df_merged['历史出现次数'].fillna(0).astype(int)

# 判定优势模型
def determine_winner(delta):
    if delta > 0.01:
        return '专属 Emb (赢)'
    elif delta < -0.01:
        return '全局 Glo (赢)'
    else:
        return '平局'

df_merged['优势模型'] = df_merged['Delta_MRR (Emb - Glo)'].apply(determine_winner)

# ================= 5. 整理并输出终极表格 =================
# 👑 核心修改：把 Alpha_Emb 和 Alpha_Glo 一起放进我们要提取的列里！
final_df = df_merged[[
    'src_id', 
    '历史出现次数', 
    'Alpha_Emb(专属)', 
    'Alpha_Glo(全局)', 
    '优势模型', 
    'Delta_MRR (Emb - Glo)'
]]

# 严格按照真实的 MRR 差值降序排列（绝不使用绝对值）
final_df = final_df.sort_values(by='Delta_MRR (Emb - Glo)', ascending=False)

# 在控制台打印前 40 行，让你直接看到专属模型大杀四方的样本
print("\n" + "="*80)
print("🏆 【专属模型碾压局】(Delta > 0) —— 看这些人的专属 Alpha 到底偏离大盘多少！")
print("="*80)
print(final_df.head(40).to_string(index=False))

# 在控制台打印后 40 行，让你直接看到全局模型兜底成功的样本
print("\n" + "="*80)
print("🛡️ 【全局模型兜底局】(Delta < 0) —— 看 Glo_Alpha 0.73 左右的威力！")
print("="*80)
print(final_df.tail(40).to_string(index=False))

# 保存成最终版文件
out_path = 'ultimate_line_by_line_analysis.csv'
final_df.to_csv(out_path, index=False)
print(f"\n✅ 终极宽表已生成！所有 {len(final_df)} 行数据已保存至: {out_path}")
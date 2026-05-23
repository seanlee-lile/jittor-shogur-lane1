import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

def make_local_dataset(data_dir, dataset, val_ratio=0.1):
    print(f"==== 开始为 {dataset} 构建本地评测集 ====")
    
    train_file = os.path.join(data_dir, dataset, 'train.csv')
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"找不到文件: {train_file}")

    # 1. 读取原始训练数据
    df = pd.read_csv(train_file)
    print(f"原始训练集总行数: {len(df)}")

    # 2. 按时间严格排序 (时序图链路预测的铁律：绝不能打乱时间穿越到未来)
    df = df.sort_values('time').reset_index(drop=True)

    # 3. 按比例切分 (前 90% 留作本地训练，后 10% 拿来做本地测试)
    split_idx = int(len(df) * (1 - val_ratio))
    local_train = df.iloc[:split_idx]
    local_test_raw = df.iloc[split_idx:]

    print(f"切分完毕: 本地训练集 {len(local_train)} 行, 本地测试集 {len(local_test_raw)} 行")

    # 4. 提取全图所有出现过的节点集合，用于给测试集抽样“假答案(负样本)”
    all_nodes = np.unique(np.concatenate([df['src'].values, df['dst'].values]))

    # 5. 构造符合你代码逻辑的 test.csv
    # 你的 eval_ranking.py 会提取第 3 列 (iloc[:, 2]) 作为正确答案
    # 所以我们必须把真实的 dst 放在候选项的第一个 (c1)
    
    test_rows = []
    print("正在为本地测试集生成 100 个候选项 (包含 1 个真答案 + 99 个假答案)...")
    
    for _, row in tqdm(local_test_raw.iterrows(), total=len(local_test_raw)):
        src = int(row['src'])
        time = int(row['time'])
        true_dst = int(row['dst'])

        # 随机抽取 99 个负样本 (必须排除掉真实的那个节点)
        negatives = []
        while len(negatives) < 99:
            # 批量抽样以提高速度
            samples = np.random.choice(all_nodes, 150, replace=False)
            valid_samples = [n for n in samples if n != true_dst]
            negatives.extend(valid_samples)
            negatives = negatives[:99] # 严格截断到 99 个

        # 【核心逻辑】真实节点打头阵，后面跟着 99 个假节点
        candidates = [true_dst] + negatives
        
        # 组装这一行的数据：src, time, c1, c2 ... c100
        test_rows.append([src, time] + candidates)

    # 生成列名：src, time, c1, c2 ... c100
    col_names = ['src', 'time'] + [f'c{i}' for i in range(1, 101)]
    local_test_df = pd.DataFrame(test_rows, columns=col_names)

    # 6. 保存到新的文件夹，防止覆盖官方数据
    out_dir = os.path.join(data_dir, f"{dataset}_local")
    os.makedirs(out_dir, exist_ok=True)

    local_train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    local_test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)

    print(f"==== 成功！数据已保存至: {out_dir} ====")
    print("现在你可以使用 --dataset dataset1_local 来运行你的 main.py 了！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='dataset1')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='切分给本地测试集的比例')
    args = parser.parse_args()

    make_local_dataset(args.data_dir, args.dataset, args.val_ratio)
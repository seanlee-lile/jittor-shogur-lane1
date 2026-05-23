import os
import shutil
from pathlib import Path

def archive_files():
    # 定义需要存档的文件/文件夹列表
    files_to_copy = [
        "main_gcn.py",
        "data/dataset1/dataset1_result.csv",
        "data/dataset2/dataset2_result.csv",
        "saved_models"
    ]

    # 定义主history文件夹路径
    history_base = Path("history")
    
    # 如果 history 文件夹不存在，则创建它
    history_base.mkdir(parents=True, exist_ok=True)

    # 自动寻找下一个可用的 n (例如 history_1, history_2...)
    n = 1
    while True:
        target_dir = history_base / f"history_{n}"
        if not target_dir.exists():
            break
        n += 1

    # 创建本次的 history_n 文件夹
    target_dir.mkdir()
    print(f"创建存档文件夹: {target_dir}")

    # 开始复制文件或文件夹
    for file_path_str in files_to_copy:
        src = Path(file_path_str)
        if src.exists():
            target_file = target_dir / src.name
            
            # 区分文件和文件夹
            if src.is_dir():
                # 如果是文件夹，使用 copytree 复制整个目录
                shutil.copytree(src, target_file)
                print(f"已复制文件夹: {src} -> {target_file}")
            else:
                # 如果是文件，保持原有的 copy2
                shutil.copy2(src, target_file)
                print(f"已复制文件: {src} -> {target_file}")
        else:
            print(f"[警告] 找不到文件或文件夹，已跳过: {src}")

    print(f"\n[完成] 所有文件已成功存入 {target_dir}")

if __name__ == "__main__":
    archive_files()
import subprocess
import sys

def run_commands():
    # 定义需要按顺序执行的命令列表
    # 使用列表形式（list）比直接用字符串更安全，可以自动处理路径中的空格
    commands = [
        ["python", "main.py", "--dataset", "dataset1_local", "--epochs", "0"],
        ["python", "main.py", "--dataset", "dataset2_local", "--epochs", "0"],
        ["python", "eval_ranking.py", "--data_dir", "data"]
    ]

    for cmd in commands:
        print(f"\n>>> 正在执行: {' '.join(cmd)}")
        try:
            # subprocess.run 会等待命令执行结束
            # check=True 表示如果命令执行失败（返回非0状态码），会抛出异常
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[错误] 命令执行失败: {e}")
            print("脚本已停止。")
            sys.exit(1)
        except FileNotFoundError:
            print(f"\n[错误] 未找到 python 或脚本文件，请确保环境配置正确。")
            sys.exit(1)

    print("\n[完成] 所有任务已成功按顺序执行完毕！")

if __name__ == "__main__":
    run_commands()
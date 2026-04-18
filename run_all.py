import os
import sys
import subprocess
import logging
import time
import argparse
from datetime import datetime

# 初始化日志功能，输出到终端和独立日志文件
log_file = f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def format_duration(seconds):
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback during training")
    return p.parse_args()


def run_script(script_name, extra_args=None):
    logging.info(f"\n{'='*50}")
    logging.info(f"开始运行阶段: {script_name}")
    logging.info(f"{'='*50}\n")

    stage_start = time.perf_counter()
    cmd = [sys.executable, script_name]
    if extra_args:
        cmd.extend(extra_args)

    # 运行对应的 Python 脚本
    result = subprocess.run(cmd)

    elapsed = time.perf_counter() - stage_start

    # 检查是否成功执行
    if result.returncode != 0:
        logging.error(f"\n{'!'*50}")
        logging.error(f"[错误] {script_name} 运行失败，退出码: {result.returncode}")
        logging.error(f"{script_name} 已运行时长: {format_duration(elapsed)}")
        logging.error(f"{'!'*50}")
        sys.exit(result.returncode) # 如果失败则直接退出，不继续执行后面的步骤
    else:
        logging.info(f"\n{'-'*50}")
        logging.info(f"阶段完成: {script_name}")
        logging.info(f"阶段耗时: {format_duration(elapsed)}")
        logging.info(f"{'-'*50}\n")
        return elapsed

if __name__ == "__main__":
    args = parse_args()
    logging.info(">>> 启动 RoadAttNet 完整流程: 特征提取 -> 模型训练 <<<\n")
    logging.info("说明: 特征提取阶段是 OpenCV/NumPy 的 CPU 流程，GPU 主要用于训练/测试/推理阶段。")

    total_start = time.perf_counter()
    shared_args = ["--config", args.config] if args.config else []

    # 1. 首先运行特征提取
    feature_elapsed = run_script("batch_extract.py", shared_args)

    # 2. 如果特征提取成功，则开始训练模型
    train_args = list(shared_args)
    if args.allow_cpu:
        train_args.append("--allow-cpu")
    train_elapsed = run_script("train.py", train_args)

    total_elapsed = time.perf_counter() - total_start

    logging.info("\n" + "="*50)
    logging.info("流程耗时汇总")
    logging.info(f"特征提取耗时: {format_duration(feature_elapsed)}")
    logging.info(f"模型训练耗时: {format_duration(train_elapsed)}")
    logging.info(f"总耗时: {format_duration(total_elapsed)}")
    logging.info("="*50)
    logging.info("恭喜，所有流程均已成功完成！")

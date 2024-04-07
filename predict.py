import argparse
import random
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.cwru import build_test
from engine import evaluate
from model import load_model
from util.mectric import ExcelWriter

# 忽略UserWarning警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
excel_writer = ExcelWriter(
    "test.xlsx",
    ["Source", "PH0", "PH1", "PH2", "PH3"],
)


def list_type(arg):
    # 解析逗号分隔的字符串，将其转换为列表
    values = arg.split(",")
    return [float(value) for value in values]


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--model_path", type=str, default=r"checkpoints\latest.pth")
    # dataset parameters
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--sample_length", type=int, default=1024)
    parser.add_argument("--sample_num", type=int, default=240)
    parser.add_argument("--rate", type=list_type, default=[0.7, 0.3])
    parser.add_argument("--feature", type=str, default="all")
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--fft", type=bool, default=True)
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    return parser


def main(args):
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # =================================================================
    # 模型
    model = load_model(args)
    # =================================================================
    # 验证
    acc_dict = {}
    acc_list = []
    for i in range(4):
        test_set = build_test(args, f"data/cwru/{i}HP")
        test_loader = DataLoader(
            dataset=test_set, batch_size=len(test_set), shuffle=False
        )
        acc = evaluate(model, test_loader, device)
        acc_dict[i] = acc
        acc_list.append(acc)
        print(f"在任务{i}HP上的测试准确率:", acc)
    acc_list.insert(0, "latest")
    excel_writer.add_row(acc_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)

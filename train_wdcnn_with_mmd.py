import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cwru import build_transfer_task, build_test
from engine import evaluate, train_one_epoch, train_one_epoch_transfer
from model import build_model
from util.criterion import create_optimizer
from util.mectric import ExcelWriter
from util.model_save import save_model

# 忽略UserWarning警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
excel_writer = ExcelWriter(
    "result/wdcnn-mmd/ac-mmd2.xlsx",
    ["Source", "T0", "T1", "T2", "T3"],
)


def list_type(arg):
    # 解析逗号分隔的字符串，将其转换为列表
    values = arg.split(",")
    return [float(value) for value in values]


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    # parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    # parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # dataset parameters
    parser.add_argument("--root", type=str, default=r"data/cwru")
    parser.add_argument("--task", type=str, default=r"0-1")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--sample_length", type=int, default=1024)
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--rate", type=list_type, default=[0.7, 0.3])
    parser.add_argument("--feature", type=str, default="de")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--fft", type=bool, default=True)

    day_time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    parser.add_argument(
        "--output_dir",
        default=f"output/{day_time}",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--save_dir",
        default="./checkpoints",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--steps",
        type=str,
        default="30, 120",
        help="the learning rate decay for step and stepLR",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="learning rate scheduler parameter for step and exp",
    )
    return parser


def main(args):
    device = torch.device(args.device)
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # =================================================================
    # 模型
    model = build_model(args)
    parameter_list = [
        {"params": model.parameters(), "lr": args.lr},
    ]
    # Define optimizer and learning rate decay
    optimizer, lr_scheduler = create_optimizer(args, parameter_list)
    criterion = nn.CrossEntropyLoss()
    # =================================================================
    # 数据
    source_train_set, source_val_set, taregt_train_set, target_val_set = (
        build_transfer_task(args)
    )
    source_train_loader = DataLoader(
        dataset=source_train_set, batch_size=args.batch_size, shuffle=True
    )
    source_val_loader = DataLoader(
        dataset=source_val_set, batch_size=len(source_val_set), shuffle=False
    )
    target_train_loader = DataLoader(
        dataset=taregt_train_set, batch_size=args.batch_size, shuffle=True
    )
    target_val_loader = DataLoader(
        dataset=target_val_set, batch_size=len(target_val_set), shuffle=False
    )
    # =================================================================
    # 开始训练
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        train_one_epoch_transfer(
            model,
            criterion,
            source_train_loader,
            target_train_loader,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
        )
        source_acc = evaluate(model, source_val_loader, device)
        target_acc = evaluate(model, target_val_loader, device)
        print("\033[0m源域验证准确率:", source_acc)
        print("\033[1;31m目标域验证准确率:", target_acc)

    # =================================================================
    # 保存
    save_model(model, args.save_dir)
    # =================================================================
    return target_acc


if __name__ == "__main__":
    for _ in range(1):
        for source_domain in range(4):
            acc_list = []
            for target_domain in range(4):
                print(f"开启任务：{source_domain}-{target_domain}")
                parser = argparse.ArgumentParser(
                    "DETR training and evaluation script", parents=[get_args_parser()]
                )
                args = parser.parse_args()
                if args.output_dir:
                    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                if args.save_dir:
                    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                args.task = f"{source_domain}-{target_domain}"
                acc = main(args)
                acc_list.append(acc)
            acc_list.insert(0, source_domain)
            excel_writer.add_row(acc_list)

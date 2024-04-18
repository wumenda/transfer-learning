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

from data.cwru import build_dataset, build_test
from engine import evaluate, train_one_epoch, evaluate_without_plot
from model.Transformer import build_model
from util.criterion import create_optimizer
from util.mectric import ExcelWriter
from util.model_save import save_model
from args_parser import get_args_parser

# 忽略UserWarning警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
excel_writer = ExcelWriter(
    "result/wdcnn/wdcnn-de/acc.xlsx",
    ["Source", "PH0", "PH1", "PH2", "PH3"],
)


def main(args):
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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
    train_set, val_set = build_dataset(args)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)
    # =================================================================
    # 开始训练
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        acc_list, losses_list = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            lr_scheduler,
        )
        # print(f"准确率：{acc_list[-1]}")
        # 验证准确率 6个图
        source_acc = evaluate_without_plot(model, val_loader, device)
        print("验证准确率:", source_acc)
    # =================================================================
    # 验证
    acc_dict = {}
    acc_list = []
    for i in range(4):
        test_set = build_test(args, f"data/cwru/{i}HP")
        test_loader = DataLoader(
            dataset=test_set, batch_size=len(test_set), shuffle=False
        )
        target_acc = evaluate(
            model,
            test_loader,
            device,
            os.path.join(args.figure_path, f"matrix-wdcnn{i}"),
            os.path.join(args.figure_path, f"tsne-wdcnn{i}"),
        )
        acc_dict[i] = target_acc
        acc_list.append(target_acc)
        print(f"在任务{i}HP上的测试准确率:", target_acc)
    acc_list.insert(0, i)
    excel_writer.add_row(acc_list)
    # with open("result.json", "a") as f:
    #     json.dump(acc_dict, f)
    #     f.write(",\n")  # 每次追加后换行
    # =================================================================
    # 保存
    save_model(model, args.save_dir)
    # =================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.figure_path = "figure/Transformer"
    args.epochs = 200
    args.fft = True
    args.lr = 1e-3
    for i in range(1):
        main(args)

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cwru import build_transfer_task
from engine import evaluate_gan, train_one_epoch_on_gan_with_mmd
from model.GAN import build_model
from util.criterion import create_optimizer
from util.mectric import ExcelWriter
from util.model_save import save_model
from util.plot import plot, plot_curve
from args_parser import get_args_parser
import numpy as np

# 忽略UserWarning警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
excel_writer = ExcelWriter(
    "result/wdcnn-mmd/ac-mmd2.xlsx",
    ["Source", "T0", "T1", "T2", "T3"],
)


def main(args):
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # =================================================================
    # 模型
    feature_net, classifier, discriminator = build_model(args)
    parameter_list = [
        {"params": feature_net.parameters(), "lr": 0.5 * args.lr},
        {"params": classifier.parameters(), "lr": 1 * args.lr},
        {"params": discriminator.parameters(), "lr": 1 * args.lr},
    ]
    # Define optimizer and learning rate decay
    optimizer, lr_scheduler = create_optimizer(args, parameter_list)
    cls_criterion = nn.CrossEntropyLoss()
    adver_criterion = nn.BCELoss()
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
    acc_lists, losses_lists, cls_loss_lists, adver_loss_lists = [], [], [], []
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        # 训练
        acc_list, losses_list, cls_loss_list, adver_loss_list = (
            train_one_epoch_on_gan_with_mmd(
                feature_net,
                classifier,
                discriminator,
                cls_criterion,
                adver_criterion,
                source_train_loader,
                target_train_loader,
                optimizer,
                device,
                epoch,
                args.clip_max_norm,
            )
        )
        acc_lists.extend(acc_list)
        losses_lists.extend(losses_list)
        cls_loss_lists.extend(cls_loss_list)
        adver_loss_lists.extend(adver_loss_list)
    # 验证准确率 6个图
    source_acc = evaluate_gan(
        feature_net,
        classifier,
        source_val_loader,
        device,
        os.path.join(args.figure_path, "source-matrix-gan-mmd-w"),
        os.path.join(args.figure_path, "source-tsne-gan-mmd-w"),
    )
    target_acc = evaluate_gan(
        feature_net,
        classifier,
        target_val_loader,
        device,
        os.path.join(args.figure_path, "target-matrix-gan-mmd-w"),
        os.path.join(args.figure_path, "target-tsne-gan-mmd-w"),
    )
    plot_curve(
        acc_lists,
        os.path.join(args.figure_path, "gan-mmd-w-acc"),
        title="accuracy ",
        xlabel="X",
        ylabel="Y",
    )
    plot_curve(
        losses_lists,
        os.path.join(args.figure_path, "gan-mmd-w-loss"),
        title="loss",
        xlabel="X",
        ylabel="Y",
    )
    print("\033[0m源域验证准确率:", source_acc)
    print("\033[1;31m目标域验证准确率:", target_acc)

    # =================================================================
    # 保存
    torch.save(
        {
            "feature_net": feature_net.state_dict(),
            "classifier": classifier.state_dict(),
        },
        os.path.join(args.save_dir, day_time + ".pth"),
    )
    # =================================================================
    return target_acc


def train_loop(loop_num=1):
    day_time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    for _ in range(loop_num):
        the_plot_acc = []
        for source_domain in range(4):
            acc_list = []
            for target_domain in range(4):
                print(f"开启任务：{source_domain}-{target_domain}")
                parser = argparse.ArgumentParser(
                    "WDCNN training and evaluation script", parents=[get_args_parser()]
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
            acc_list.pop(0)
            the_plot_acc.append(acc_list)
        plot(the_plot_acc, f"acc-{day_time}.png")


if __name__ == "__main__":
    day_time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    parser = argparse.ArgumentParser(
        "WDCNN training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.figure_path = "figure/gan-mmd-w"
    args.acc_result = "result/gan-mmd-w"
    args.lr = 1e-3
    args.batch_size = 512
    args.epochs = 400
    args.fft = True
    args.rate = [0.7, 0.3]
    acc = main(args)
    with open(os.path.join(args.acc_result, "acc.txt"), "a") as f:
        # 重定向 print 的输出到文件
        print(f"{acc}", file=f)

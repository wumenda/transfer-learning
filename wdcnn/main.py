import torch
import wdcnn.preprocess as preprocess
from torch import nn
import train1 as tr
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cwru import build_dataset
from engine1 import evaluate, train_one_epoch
from WDCNN import build_model
from util.criterion import create_optimizer
from util.model_save import save_model

# 忽略UserWarning警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 训练参数
batch_size = 128
epochs = 20
num_classer = 10
length = 1024  # 采样长度
BatchNorm = True  # 是否批量归一化
number = 100  # 每类样本数量
normal = False  # 是否对样本标准化
rate = [0.7, 0.2, 0.1]  # 训练集、测试集、样本集划分比例

path = "data\\cwru\\0HP"
train_X, train_Y, test_X, test_Y, valid_X, valid_Y = preprocess.prepro(
    data_path=path,
    length=length,
    number=number,
    slice_rate=rate,
    enc=True,
    enc_step=28,
    normal=normal,
)

# 把数据从array个数转成tensor格式
# 把列信号看作一个length * 1的二维“图片”，然后就可以用卷积了
# 但是torch框架里有一维卷积，所有不需要使用二维卷积来实现
# 一维卷积里数据只需要三个维度，（样本数/batchsize， 输入通道数， 样本长度）
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(
    train_Y
).float()  # 将onehot标签数据类型转换成torch.float类型，以避免在计算loss时因为数据类型不匹配报错
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).float()
valid_X = torch.from_numpy(valid_X).float()
valid_Y = torch.from_numpy(valid_Y).float()
# 将样本数据增加维度变成三维张量，torch里的一维卷积是对三维张量进行
train_X = train_X[:, None, :]
test_X = test_X[:, None, :]
valid_X = valid_X[:, None, :]

# 将数据切分为随机batch_size
train_iter = tr.slice_to_batch_size(
    train_X, train_Y, batch_size=batch_size, shuffle=True
)
test_iter = tr.slice_to_batch_size(test_X, test_Y, batch_size=batch_size, shuffle=True)


def list_type(arg):
    # 解析逗号分隔的字符串，将其转换为列表
    values = arg.split(",")
    return [float(value) for value in values]


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    # parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    # parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # dataset parameters
    parser.add_argument("--root", type=str, default=r"data/cwru/0HP")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--sample_length", type=int, default=2048)
    parser.add_argument("--sample_num", type=int, default=400)
    parser.add_argument("--rate", type=list_type, default=[0.7, 0.3])

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
        "--device", default="cpu", help="device to use for training / testing"
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
    # 随机数种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 模型
    model = build_model(args)

    parameter_list = [
        {"params": model.parameters(), "lr": args.lr},
    ]

    # Define optimizer and learning rate decay
    optimizer, lr_scheduler = create_optimizer(args, parameter_list)

    criterion = (
        nn.CrossEntropyLoss()
    )  # 设置交叉熵误差,默认计算的是输入的一个batch的平均误差

    # 数据
    train_set, test_set = build_dataset(args)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=False)

    # 开始训练
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        # 训练
        train_one_epoch(
            model,
            criterion,
            train_iter,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            lr_scheduler,
        )

    # 验证
    acc = evaluate(model, test_iter, device)
    # 保存
    save_model(model, args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    main(args)

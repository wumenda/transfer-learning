import argparse
from datetime import datetime


def list_type(arg):
    # 解析逗号分隔的字符串，将其转换为列表
    values = arg.split(",")
    return [float(value) for value in values]


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    # parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    # parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=r"latest.path",
    )

    # dataset parameters
    parser.add_argument("--root", type=str, default=r"data/cwru")
    parser.add_argument("--task", type=str, default=r"0-3")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--sample_length", type=int, default=1024)
    parser.add_argument("--sample_num", type=int, default=240)
    parser.add_argument("--rate", type=float, default=0.7)
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
        "--acc_result",
        default=f"result",
        help="path where to save",
    )
    parser.add_argument(
        "--figure_path",
        default=f"figure/gan",
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
        default="150, 900",
        help="the learning rate decay for step and stepLR",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="learning rate scheduler parameter for step and exp",
    )
    return parser

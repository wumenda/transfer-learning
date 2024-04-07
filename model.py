import torch
import torch.nn as nn


class Wdcnn(nn.Module):
    def __init__(self, num_classes, input_size, channels):
        super(Wdcnn, self).__init__()
        self.cbrp1 = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cbrp2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cbrp3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cbrp4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cbrp5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size // 8, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.cbrp1(x)
        x = self.cbrp2(x)
        x = self.cbrp3(x)
        x = self.cbrp4(x)
        x = self.cbrp5(x)
        feature = self.mlp(x)
        out = self.head(feature)
        return feature, out


def init_weights(m):
    """定义网络的参数初始化方法"""
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)


def build_model(args):
    device = torch.device(args.device)
    model = Wdcnn(args.num_classes, args.sample_length, args.channels)
    model.apply(init_weights)  # 对网络进行初始化
    model.to(device)
    return model


def load_model(args):
    """
    加载模型并加载本地模型参数。

    参数:
    - model_path (str): 模型文件的路径。
    - num_classes (int): 类别数。
    - input_size (int): 输入数据的大小。
    - channels (int): 输入数据的通道数。

    返回:
    - model: 加载了参数的模型。
    """
    device = torch.device(args.device)
    # 创建模型实例
    model = Wdcnn(args.num_classes, args.sample_length, args.channels)

    # 加载模型参数
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 将模型设置为评估模式
    model.eval()
    model.to(device)

    return model

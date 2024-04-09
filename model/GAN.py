import torch
import torch.nn as nn
import numpy as np


# ===== Build Model =====
class CnnFeatureNet(nn.Module):
    def __init__(self, input_size, channels):
        super(CnnFeatureNet, self).__init__()
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

    def forward(self, x):
        x = self.cbrp1(x)
        x = self.cbrp2(x)
        x = self.cbrp3(x)
        x = self.cbrp4(x)
        x = self.cbrp5(x)
        logits = self.mlp(x)
        return logits


# ===== Build Model =====
class TransformerFeatureNet(nn.Module):
    def __init__(
        self,
        input_size,
        channels,
        transformer_layers=3,
        transformer_dim=64,
        transformer_heads=8,
    ):
        super(TransformerFeatureNet, self).__init__()
        self.embedding = nn.Linear(input_size, transformer_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=transformer_layers
        )
        self.mlp = nn.Sequential(nn.Linear(transformer_dim, 64), nn.ReLU())

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(
            1, 0, 2
        )  # Transformer要求输入形状为(seq_length, batch_size, embedding_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # 取所有时间步的平均作为整个序列的表示
        logits = self.mlp(x)
        return logits


class Classifier(nn.Module):
    def __init__(self, num_out=10):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(nn.Linear(64, num_out, nn.Dropout(0.5)))

    def forward(self, logits):
        outputs = self.head(logits)
        return outputs


class Discriminator(nn.Module):
    def __init__(
        self,
        num_out=1,
        max_iter=10000.0,
    ):
        super(Discriminator, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 128, nn.Dropout(0.5)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_out),
        )

        self.sigmoid = nn.Sigmoid()

        # parameters
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1
            coeff = calc_coeff(
                self.iter_num, self.high, self.low, self.alpha, self.max_iter
            )
        else:
            raise Exception("loss not implement")
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.domain_classifier(x)
        x = self.sigmoid(x)
        return x


# 权重衰减
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float64(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


# 梯度反转模块
def grl_hook(coeff):
    def gradient_reverse(grad):
        return -coeff * grad.clone()

    return gradient_reverse


def init_weights(m):
    """定义网络的参数初始化方法"""
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)


def build_model(args):
    device = torch.device(args.device)
    feature_net = CnnFeatureNet(args.sample_length, args.channels)
    classifier = Classifier(
        args.num_classes,
    )
    discriminator = Discriminator()
    feature_net.apply(init_weights)  # 对网络进行初始化
    feature_net.to(device)
    classifier.apply(init_weights)  # 对网络进行初始化
    classifier.to(device)
    discriminator.apply(init_weights)  # 对网络进行初始化
    discriminator.to(device)
    return feature_net, classifier, discriminator

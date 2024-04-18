import torch
import torch.nn as nn
import numpy as np
import math


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
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.cbrp1(x)
        x = self.cbrp2(x)
        x = self.cbrp3(x)
        x = self.cbrp4(x)
        x = self.cbrp5(x)
        logits = self.mlp(x)
        return logits


class TransformerNet(nn.Module):
    def __init__(
        self,
        input_size,
        channels,
        transformer_layers=4,
        transformer_dim=128,
        transformer_heads=8,
    ):
        super(TransformerNet, self).__init__()

        self.transformer_dim = transformer_dim
        self.positional_encoding = self.generate_positional_encoding(
            input_size, transformer_dim
        )
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, dim_feedforward=256, nhead=transformer_heads
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=transformer_layers
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.transformer_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Sigmoid(),
        )

        # 添加一个新的向量作为输入
        self.new_vector = nn.Parameter(torch.zeros(1, 1, transformer_dim))

    def forward(self, x):
        batch_size, channels, length = x.size()
        seq_length = length // self.transformer_dim
        x = x.view(
            batch_size, seq_length, self.transformer_dim
        )  # 调整输入维度顺序以适应Transformer
        # 在输入序列的最前面添加新的向量
        new_vector = self.new_vector.expand(batch_size, 1, self.transformer_dim)
        x = torch.cat([new_vector, x], dim=1)
        x = x + self.positional_encoding[: seq_length + 1, :].unsqueeze(0).expand(
            batch_size, -1, -1
        )  # 添加位置编码
        x = x.permute(
            1, 0, 2
        )  # Transformer要求输入形状为(seq_length, batch_size, embedding_dim)
        y = self.transformer(x)
        feature_vector = y[0]
        logits = self.mlp(feature_vector)
        return logits

    def generate_positional_encoding(self, input_size, d_model):
        position = torch.arange(0, input_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(1000.0) / d_model)
        )
        positional_encoding = torch.zeros(input_size, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.to("cuda")


class TransformerWithConv(nn.Module):
    def __init__(
        self,
        input_size,
        channels,
    ):
        super(TransformerWithConv, self).__init__()

        self.conv = CnnFeatureNet(input_size, channels)
        self.mlp_c = nn.Linear(64, 32)
        self.transformer = TransformerNet(input_size, channels)
        self.mlp_t = nn.Linear(64, 32)
        # self.head = nn.Sequential(
        #     nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes)
        # )

    def forward(self, x):
        feature_cnn = self.conv(x)
        feature_cnn = self.mlp_c(feature_cnn)
        feature_transformer = self.transformer(x)
        feature_transformer = self.mlp_t(feature_transformer)
        logits = torch.cat([feature_cnn, feature_transformer], dim=-1)
        # output = self.head(logits)
        return logits

    def generate_positional_encoding(self, input_size, d_model):
        position = torch.arange(0, input_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(100.0) / d_model))
        positional_encoding = torch.zeros(input_size, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.to("cuda")


class Classifier(nn.Module):
    def __init__(self, num_out=10):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(nn.Linear(64, num_out))

    def forward(self, logits):
        outputs = self.head(logits)
        return outputs


class MergeClassifier(nn.Module):
    def __init__(self, num_out=10):
        super(MergeClassifier, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_out))

    def forward(self, logits):
        logits = self.sigmoid(logits)
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
            nn.Linear(128, 128, nn.Dropout(0.5)),
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
    cnn_feature_net = CnnFeatureNet(args.sample_length, args.channels)
    transformer_feature_net = TransformerNet(args.sample_length, args.channels)
    classifier4cnn = Classifier(
        args.num_classes,
    )
    classifier4transformer = Classifier(
        args.num_classes,
    )
    classifier4merge = MergeClassifier(
        args.num_classes,
    )
    discriminator = Discriminator()
    cnn_feature_net.apply(init_weights)  # 对网络进行初始化
    cnn_feature_net.to(device)
    transformer_feature_net.apply(init_weights)  # 对网络进行初始化
    transformer_feature_net.to(device)
    cnn_feature_net.apply(init_weights)  # 对网络进行初始化
    cnn_feature_net.to(device)
    classifier4cnn.apply(init_weights)  # 对网络进行初始化
    classifier4cnn.to(device)
    classifier4transformer.apply(init_weights)  # 对网络进行初始化
    classifier4transformer.to(device)
    classifier4merge.apply(init_weights)  # 对网络进行初始化
    classifier4merge.to(device)
    discriminator.apply(init_weights)  # 对网络进行初始化
    discriminator.to(device)
    return (
        cnn_feature_net,
        transformer_feature_net,
        classifier4cnn,
        classifier4transformer,
        classifier4merge,
        discriminator,
    )


# def load_model(args):
#     device = args.device
#     # Initialize network instances
#     feature_net = CnnFeatureNet(args.sample_length, args.channels)
#     classifier = Classifier(args.num_classes)

#     # Load saved model parameters
#     checkpoint = torch.load(args.model_path)
#     feature_net.load_state_dict(checkpoint["feature_net"])
#     classifier.load_state_dict(checkpoint["classifier"])

#     # Move models to the desired device
#     feature_net.to(device)
#     classifier.to(device)

#     return feature_net, classifier

import torch
import torch.nn as nn
import math
from .WDCNN import Wdcnn


class TransformerNet(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        channels,
        transformer_layers=4,
        transformer_dim=128,
        transformer_heads=8,
    ):
        super(TransformerNet, self).__init__()
        self.transformer_dim = transformer_dim
        self.positional_encoding = self.generate_positional_encoding(
            input_size // transformer_dim, transformer_dim
        )
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, dim_feedforward=512, nhead=transformer_heads
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=transformer_layers
        )
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU()
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, channels, length = x.size()
        seq_length = length // self.transformer_dim
        x = x.view(
            batch_size, seq_length, self.transformer_dim
        )  # 调整输入维度顺序以适应Transformer
        x = x + self.positional_encoding[:seq_length, :].unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # 添加位置编码
        x = x.permute(
            1, 0, 2
        )  # Transformer要求输入形状为(seq_length, batch_size, embedding_dim)
        y = self.transformer(x)
        x = y.permute(1, 0, 2).contiguous().view(batch_size, -1)
        logits = self.mlp(x)
        output = self.head(logits)
        return logits, output

    def generate_positional_encoding(self, input_size, d_model):
        position = torch.arange(0, input_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(1000.0) / d_model)
        )
        positional_encoding = torch.zeros(input_size, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.to("cuda")


class Transformer(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        channels,
        transformer_layers=4,
        transformer_dim=128,
        transformer_heads=8,
    ):
        super(Transformer, self).__init__()

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
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.2)  # 添加 Dropout 层

        # 添加一个新的向量作为输入
        self.new_vector = nn.Parameter(torch.ones(1, 1, transformer_dim))

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
        # x = self.dropout(x)  # 应用 Dropout
        y = self.transformer(x)
        feature_vector = y[0]
        logits = self.mlp(feature_vector)
        output = self.head(logits)
        return logits, output

    def generate_positional_encoding(self, input_size, d_model):
        position = torch.arange(0, input_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(100.0) / d_model))
        positional_encoding = torch.zeros(input_size, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.to("cuda")


class TransformerWithConv(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        channels,
    ):
        super(TransformerWithConv, self).__init__()

        self.conv = Wdcnn(num_classes, input_size, channels)
        self.mlp_c = nn.Linear(32, 16)
        self.transformer = Transformer(num_classes, input_size, channels)
        self.mlp_t = nn.Linear(32, 16)
        self.head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes)
        )

    def forward(self, x):
        feature_cnn, _ = self.conv(x)
        feature_cnn = self.mlp_c(feature_cnn)
        feature_transformer, _ = self.transformer(x)
        feature_transformer = self.mlp_t(feature_transformer)
        logits = torch.cat([feature_cnn, feature_transformer], dim=-1)
        output = self.head(logits)
        return logits, output

    def generate_positional_encoding(self, input_size, d_model):
        position = torch.arange(0, input_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(100.0) / d_model))
        positional_encoding = torch.zeros(input_size, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.to("cuda")


def init_weights(m):
    """定义网络的参数初始化方法"""
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)


def build_model(args):
    device = torch.device(args.device)
    model = Transformer(args.num_classes, args.sample_length, args.channels)
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
    model = Transformer(args.num_classes, args.sample_length, args.channels)

    # 加载模型参数
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 将模型设置为评估模式
    model.eval()
    model.to(device)

    return model

import torch
import torch.nn as nn
import torchvision.models as models
import math


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
        num_classes,
        input_size,
        channels,
    ):
        super(TransformerWithConv, self).__init__()

        self.conv = CnnFeatureNet(input_size, channels)
        self.transformer = TransformerNet(input_size, channels)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feature_cnn = self.conv(x)
        feature_transformer = self.transformer(x)
        logits = torch.cat([feature_cnn, feature_transformer], dim=-1)
        output = self.head(logits)
        return logits, output


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
        self.dropout = nn.Dropout(p=0.5)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size // 8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.cbrp1(x)
        x = self.cbrp2(x)
        x = self.cbrp3(x)
        x = self.cbrp4(x)
        x = self.cbrp5(x)
        feature = self.mlp(x)
        out = self.head(feature)
        return feature, out


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        # 加载预训练的ResNet-18模型，不包含分类层
        self.resnet = models.resnet18(pretrained=True)
        # 替换最后一层的全连接层，使输出维度与类别数相匹配
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class Bottlrneck(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class ResNet(torch.nn.Module):

    def __init__(self, classes, input_size, in_channels=1):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),
            Bottlrneck(64, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            #
            Bottlrneck(256, 128, 512, True),
            Bottlrneck(512, 128, 512, False),
            Bottlrneck(512, 128, 512, False),
            Bottlrneck(512, 128, 512, False),
            #
            Bottlrneck(512, 256, 1024, True),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            #
            Bottlrneck(1024, 512, 2048, True),
            Bottlrneck(2048, 512, 2048, False),
            Bottlrneck(2048, 512, 2048, False),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifer = torch.nn.Sequential(torch.nn.Linear(2048, classes))

    def forward(self, x):
        x = self.features(x)
        features = x.view(-1, 2048)
        out = self.classifer(features)
        return features, out


class VGG19(torch.nn.Module):

    def __init__(self, classes, input_size, in_channels=1):
        super(VGG19, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.AdaptiveAvgPool1d(7),
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(3584, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, classes),
        )

    def forward(self, x):
        feature = self.feature(x)
        x = feature.view(-1, 3584)
        out = self.classifer(x)
        return feature, out


class Transformer(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        channels,
        transformer_layers=3,
        transformer_dim=64,
        transformer_heads=4,
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_size, transformer_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=transformer_layers
        )
        self.mlp = nn.Sequential(nn.Linear(transformer_dim, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(
            1, 0, 2
        )  # Transformer要求输入形状为(seq_length, batch_size, embedding_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # 取所有时间步的平均作为整个序列的表示
        logits = self.mlp(x)
        outputs = self.head(logits)
        return logits, outputs


class WdcnnWithExtraFeature(nn.Module):
    def __init__(self, num_classes, input_size, channels, extra_feature_size):
        super(WdcnnWithExtraFeature, self).__init__()
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
        # Additional linear layer for manually extracted features
        self.extra_linear = nn.Linear(
            extra_feature_size, 64
        )  # Adjust extra_feature_size as needed
        self.head = nn.Sequential(
            nn.Linear(128, num_classes)
        )  # Concatenating with extra features

    def forward(self, x, extra_features):
        x = self.cbrp1(x)
        x = self.cbrp2(x)
        x = self.cbrp3(x)
        x = self.cbrp4(x)
        x = self.cbrp5(x)
        feature = self.mlp(x)
        # Process manually extracted features
        extra_features = self.extra_linear(extra_features)
        # Concatenate extracted features with the CNN features
        concatenated_features = torch.cat((feature, extra_features), dim=1)
        out = self.head(concatenated_features)
        return concatenated_features, out


def init_weights(m):
    """定义网络的参数初始化方法"""
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)


def build_model(args):
    device = torch.device(args.device)
    model = TransformerWithConv(args.num_classes, args.sample_length, args.channels)
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

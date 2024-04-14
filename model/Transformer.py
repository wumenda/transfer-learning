import torch
import torch.nn as nn


class WdcnnTransformer(nn.Module):
    def __init__(self, num_classes, input_size, channels):
        super(WdcnnTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=5
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_size // 8, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将输入调整为(batch_size, channels, seq_len)的格式
        x = self.transformer_encoder(x)
        feature = self.mlp(x[:, :, -1])  # 取最后一个时间步的输出作为特征表示
        out = self.head(feature)
        return feature, out

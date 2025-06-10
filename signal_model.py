import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class SelfAttention1D(nn.Module):
    def __init__(self, dim):
        super(SelfAttention1D, self).__init__()
        self.query = nn.Conv1d(dim, dim, 1)
        self.key = nn.Conv1d(dim, dim, 1)
        self.value = nn.Conv1d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = self.softmax(torch.bmm(Q.transpose(1, 2), K) / (Q.size(1) ** 0.5))
        out = torch.bmm(attn, V.transpose(1, 2)).transpose(1, 2)
        return out + x  # Residual

class SignalEstimatorNet(nn.Module):
    def __init__(self, input_len=300):
        super(SignalEstimatorNet, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock1D(1, 16),
            ResidualBlock1D(16, 32),
            ResidualBlock1D(32, 64),
            nn.AdaptiveAvgPool1d(input_len)
        )

        self.attn = SelfAttention1D(64)

        self.ppg_head = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )

        self.ecg_head = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.attn(x)
        ppg = self.ppg_head(x).squeeze(1)  # (B, L)
        ecg = self.ecg_head(x).squeeze(1)
        return ppg, ecg


import torch
import torch.nn as nn

class rPPGEstimatorCNN(nn.Module):
    def __init__(self, output_length=300):
        super(rPPGEstimatorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, output_length)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

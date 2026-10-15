"""CNN model for Tibetan character recognition.

CPU-optimized architecture based on deep learning best practices:
- Depthwise separable convolutions (~8-18x fewer operations)
- Batch normalization (faster convergence, higher learning rates)
- Residual connections (deeper networks without vanishing gradients)
- Global Average Pooling (eliminates millions of FC parameters)
- He initialization for ReLU networks

Input: 32x32 grayscale character images (1 channel)
Output: num_classes logits
"""

import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise.

    Factored convolution that reduces computation by ~8-18x compared
    to standard convolution for 3x3 kernels.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            stride=stride, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with two depthwise separable convolutions.

    Skip connection adds input directly to output, enabling
    training of deeper networks without vanishing gradients.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        # Second conv without final ReLU (applied after residual add)
        self.conv2_dw = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.conv2_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv2_pw(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class TibetanCNN(nn.Module):
    """CPU-optimized CNN for Tibetan character classification.

    Architecture:
        Input (1x32x32)
        -> Conv2d(32, 3x3) + BN + ReLU              [32x32x32]   stem
        -> DSConv(32->64, stride=2)                   [64x16x16]  downsample
        -> ResidualBlock(64)                          [64x16x16]
        -> DSConv(64->128, stride=2)                  [128x8x8]   downsample
        -> ResidualBlock(128)                         [128x8x8]
        -> DSConv(128->256, stride=2)                 [256x4x4]   downsample
        -> Global Average Pooling                     [256]
        -> Dropout
        -> FC(256, num_classes)

    ~350K parameters. Inference: <5ms per character on CPU.
    """

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        # Stem: standard conv to learn from raw pixels
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Downsample 32x32 -> 16x16, expand 32 -> 64
        self.down1 = DepthwiseSeparableConv(32, 64, stride=2)
        self.res1 = ResidualBlock(64)

        # Downsample 16x16 -> 8x8, expand 64 -> 128
        self.down2 = DepthwiseSeparableConv(64, 128, stride=2)
        self.res2 = ResidualBlock(128)

        # Downsample 8x8 -> 4x4, expand 128 -> 256
        self.down3 = DepthwiseSeparableConv(128, 256, stride=2)

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        """He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)       # [B, 32, 32, 32]
        x = self.down1(x)      # [B, 64, 16, 16]
        x = self.res1(x)       # [B, 64, 16, 16]
        x = self.down2(x)      # [B, 128, 8, 8]
        x = self.res2(x)       # [B, 128, 8, 8]
        x = self.down3(x)      # [B, 256, 4, 4]
        x = self.gap(x)        # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        x = self.dropout(x)
        x = self.fc(x)         # [B, num_classes]
        return x

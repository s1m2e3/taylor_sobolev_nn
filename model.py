import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if isinstance(self.shortcut, nn.Conv2d) else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class model(nn.Module):
    """
    A small ResNet-style model for image classification on 32x32 images.

    This model aims to provide universal approximation capabilities with a
    significantly reduced number of parameters compared to larger models like ResNet18.
    It uses residual blocks.
    """
    def __init__(self, pixel_size, in_channels=3, num_classes=10):
        super(model, self).__init__()
        # For CIFAR-10, the initial 7x7 conv and maxpool in standard ResNet
        # are often replaced with a single 3x3 conv.
        self.in_channels = pixel_size
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        
        # Residual blocks
        self.layer1 = self._make_layer(self.in_channels, 1, stride=1)
        self.layer2 = self._make_layer(self.in_channels*2, 1, stride=2)
        self.layer3 = self._make_layer(self.in_channels*4, 1, stride=2)

        # The input to fc is the number of channels after the last residual block
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(pixel_size * 8, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(PreActResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)


    def forward(self, x):
        # Input: (N, C, 32, 32)
        out = F.relu(self.bn1(self.conv1(x))) # -> (N, 32, 32, 32)
        
        out = self.layer1(out) # -> (N, 32, 32, 32)
        out = self.layer2(out) # -> (N, 64, 16, 16)
        out = self.layer3(out) # -> (N, 128, 8, 8)
        
        out = self.avgpool(out) # -> (N, 128, 1, 1)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out
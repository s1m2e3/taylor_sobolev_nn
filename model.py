import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    """
    A small Convolutional Neural Network (CNN) designed for image classification,
    specifically for 32x32x3 input images (like CIFAR-10).

    This model aims to provide universal approximation capabilities with a
    significantly reduced number of parameters compared to larger models like ResNet18.
    It includes convolutional layers, batch normalization, ReLU activations,
    max pooling, and fully connected layers.
    """
    def __init__(self, pixel_size,in_channels=3, num_classes=10):
        super(model, self).__init__()
        # Input: (N, 3, 32, 32) after permuting from (N, 32, 32, 3)
        self.conv1 = nn.Conv2d(in_channels, pixel_size, kernel_size=3, padding=1) # Output: (N, 32, 32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)     # Output: (N, 32, 16, 16)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Output: (N, 64, 16, 16)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)     # Output: (N, 64, 8, 8)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Output: (N, 128, 8, 8)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)     # Output: (N, 128, 4, 4)

        # Calculate the size of the flattened features before the first fully connected layer
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input is now expected to be in (N, C, H, W) format
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.reshape(-1, 128 * 4 * 4) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
from typing import Tuple, Optional


class AtrousConvBlock(nn.Module):
    """Atrous (dilated) convolution block with batch normalization and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, atrous_rate: int) -> None:
        """Initialize the AtrousConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            atrous_rate (int): Dilation rate for atrous convolution.

        Raises:
            ValueError: If in_channels or out_channels are not positive.
        """
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive integers")
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=atrous_rate,
            dilation=atrous_rate,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply atrous convolution, batch normalization, and ReLU activation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module for multi-scale feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5) -> None:
        """Initialize the ASPP module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels per branch.
            dropout (float, optional): Dropout probability for the output layer. Defaults to 0.5.

        Raises:
            ValueError: If in_channels or out_channels are not positive.
        """
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive integers")

        # 1x1 Convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 Convolution branches with different atrous rates
        self.branch2 = AtrousConvBlock(in_channels, out_channels, atrous_rate=6)
        self.branch3 = AtrousConvBlock(in_channels, out_channels, atrous_rate=12)
        self.branch4 = AtrousConvBlock(in_channels, out_channels, atrous_rate=18)

        # Global Average Pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through ASPP branches and combine results.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        size = x.shape[2:]

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        # Global Average Pooling with interpolation
        out5 = self.gap(x)
        out5 = F.interpolate(out5, size=size, mode='bilinear', align_corners=True)

        # Concatenate and process outputs
        merged = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return self.conv_out(merged)


class DeepLabV3Decoder(nn.Module):
    """Decoder module for DeepLabV3+ combining low-level and ASPP features."""

    def __init__(self, low_level_in_channels: int, aspp_out_channels: int, num_classes: int) -> None:
        """Initialize the DeepLabV3Decoder.

        Args:
            low_level_in_channels (int): Number of input channels for low-level features.
            aspp_out_channels (int): Number of input channels from ASPP output.
            num_classes (int): Number of output classes for segmentation.

        Raises:
            ValueError: If any input channel count or num_classes is not positive.
        """
        super().__init__()
        if any(c <= 0 for c in [low_level_in_channels, aspp_out_channels, num_classes]):
            raise ValueError("Channel counts and num_classes must be positive")

        # Reduce low-level features
        self.low_level_reduce = nn.Sequential(
            nn.Conv2d(low_level_in_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(48 + aspp_out_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, aspp_out: torch.Tensor, low_level_features: torch.Tensor) -> torch.Tensor:
        """Combine ASPP and low-level features to produce segmentation output.

        Args:
            aspp_out (torch.Tensor): ASPP output tensor of shape (batch_size, aspp_out_channels, height, width).
            low_level_features (torch.Tensor): Low-level feature tensor of shape
                (batch_size, low_level_in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        low_level_features = self.low_level_reduce(low_level_features)
        aspp_out = F.interpolate(
            aspp_out,
            size=low_level_features.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        merged = torch.cat([low_level_features, aspp_out], dim=1)
        return self.decoder(merged)


class ResNetBackbone(nn.Module):
    """ResNet101 backbone modified for DeepLabV3+ with atrous convolutions."""

    def __init__(self, weights: Optional[ResNet101_Weights] = None) -> None:
        """Initialize the ResNet101 backbone.

        Args:
            weights (Optional[ResNet101_Weights], optional): Pretrained weights for ResNet101.
                Defaults to None.
        """
        super().__init__()
        resnet = resnet101(weights=weights)
        self.low_level_channels = 256
        self.high_level_channels = 2048

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Modify layer4 for atrous convolution
        for name, module in self.layer4.named_modules():
            if 'conv2' in name:
                module.dilation = (2, 2)
                module.padding = (2, 2)
                module.stride = (1, 1)
            elif 'downsample.0' in name:
                module.stride = (1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract high-level and low-level features from the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: High-level features of shape
                (batch_size, high_level_channels, height, width) and low-level features of shape
                (batch_size, low_level_channels, height, width).
        """
        x = self.layer0(x)
        low_level = self.layer1(x)
        high_level = self.layer2(low_level)
        high_level = self.layer3(high_level)
        high_level = self.layer4(high_level)
        return high_level, low_level


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ model for semantic segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3) -> None:
        """Initialize the DeepLabV3+ model.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1.
            num_classes (int, optional): Number of output classes. Defaults to 3.

        Raises:
            ValueError: If in_channels or num_classes are not positive.
        """
        super().__init__()
        if in_channels <= 0 or num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive")

        self.backbone = ResNetBackbone()
        self.aspp = ASPP(self.backbone.high_level_channels, 256)
        self.decoder = DeepLabV3Decoder(self.backbone.low_level_channels, 256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform semantic segmentation on the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        size = x.shape[2:]
        high_level_features, low_level_features = self.backbone(x)
        aspp_out = self.aspp(high_level_features)
        decoder_out = self.decoder(aspp_out, low_level_features)
        return F.interpolate(decoder_out, size=size, mode='bilinear', align_corners=True)
import torch
import torch.nn as nn
from .resnext3d import SEBlock3D

class MBConvBlock3D(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block for 3D EfficientNet."""
    def __init__(self, in_channels, out_channels, expansion_ratio=6, stride=1, use_se=True, kernel_size=3, reduction_ratio_se=4):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expansion_ratio)
        padding = (kernel_size - 1) // 2

        layers = []
        # expansion phase
        if expansion_ratio != 1:
            layers.append(nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # depthwise conv
        layers.append(nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm3d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # squeeze and excitation
        if use_se:
            layers.append(SEBlock3D(hidden_dim, reduction=reduction_ratio_se))

        # projection phase
        layers.append(nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm3d(out_channels))

        self.mbconv = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.mbconv(x)
        if self.use_residual:
            out += identity
        return out
    
class EfficientNet3D(nn.Module):
    """EfficientNet-B0 3D implementation."""
    def __init__(self, num_demographics, initial_filters=32, width_coefficient=1.0, depth_coefficient=1.0, filters_multiplier=1.2, use_dropout=True, use_demographics=False):
        super().__init__()
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.filters_multiplier = filters_multiplier
        self.use_dropout = use_dropout
        self.initial_filters = initial_filters
        self.use_demographics = use_demographics

        self.initial_filters = int(initial_filters * width_coefficient)
        self.last_filters = int(1280 * width_coefficient) # standard EfficientNet last filter size

        self.conv_stem = nn.Conv3d(1, self.initial_filters, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm3d(self.initial_filters)
        self.relu_stem = nn.ReLU6(inplace=True)

        # define MBConv block configurations (simplified B0-like) - tuples of (out_channels, expansion_ratio, num_layers, stride, kernel_size, use_se)
        blocks_config = [
            (int(16 * width_coefficient * filters_multiplier), 1, int(1 * depth_coefficient), 1, 3, True), # stage 1 - increased filters
            (int(24 * width_coefficient * filters_multiplier), 6, int(2 * depth_coefficient), 2, 3, True), # stage 2 - increased filters
            (int(40 * width_coefficient * filters_multiplier), 6, int(2 * depth_coefficient), 2, 5, True), # stage 3 - increased filters
            (int(80 * width_coefficient * filters_multiplier), 6, int(3 * depth_coefficient), 2, 3, True), # stage 4 - increased filters
            (int(112 * width_coefficient * filters_multiplier), 6, int(3 * depth_coefficient), 1, 5, True), # stage 5 - increased filters
            (int(192 * width_coefficient * filters_multiplier), 6, int(4 * depth_coefficient), 2, 5, True), # stage 6 - increased filters
            (int(320 * width_coefficient * filters_multiplier), 6, int(1 * depth_coefficient), 1, 3, True)  # stage 7 - increased filters
        ]
        self.blocks = nn.ModuleList()
        in_channels = self.initial_filters
        for out_channels, expansion_ratio, num_layers, stride, kernel_size, use_se in blocks_config:
            out_channels = int(out_channels)
            for i in range(num_layers):
                block_stride = stride if i == 0 else 1
                self.blocks.append(MBConvBlock3D(in_channels, out_channels, expansion_ratio, block_stride, use_se, kernel_size))
                in_channels = out_channels

        self.conv_head = nn.Conv3d(in_channels, self.last_filters, kernel_size=1, bias=False)
        self.bn_head = nn.BatchNorm3d(self.last_filters)
        self.relu_head = nn.ReLU6(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout_layer = nn.Dropout(0.2) if use_dropout and use_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.last_filters, 128)
        fc_out_size = 128 + num_demographics if self.use_demographics else 128
        self.fc_out = nn.Linear(fc_out_size, 1)
    
    def forward(self, x, demographics):
        x = self.relu_stem(self.bn_stem(self.conv_stem(x)))
        for block in self.blocks:
            x = block(x)
        x = self.relu_head(self.bn_head(self.conv_head(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_layer(x)
        x = self.fc(x)
        if self.use_demographics:
            x = torch.cat((x, demographics), dim=1)
        x = self.fc_out(x)
        return x

    def get_name(self):
        """Dynamically generate model name based on parameters."""
        name = "EfficientNet3D"
        name += f"_wc{self.width_coefficient}_dc{self.depth_coefficient}_filters{self.initial_filters}_filtermult{self.filters_multiplier}"
        if self.use_dropout > 0:
            name += "_DO"
        name += f"_dropout{self.use_dropout}"
        if self.use_demographics:
            name += "_with_demographics"
        else:
            name += "_without_demographics"
        return name

    def get_params(self):
        """Return model parameters for wandb config."""
        return {
            "width_coefficient": self.width_coefficient,
            "depth_coefficient": self.depth_coefficient,
            "initial_filters": self.initial_filters,
            "filters_multiplier": self.filters_multiplier,
            "use_dropout": self.use_dropout,
            "use_demographics": self.use_demographics,
            "architecture": "EfficientNet3D"
        }

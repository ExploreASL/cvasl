import torch
import torch.nn as nn
from .resnext3d import SEBlock3D


class ResNet3DBlock(nn.Module):
    """
    ResNet3D Block with adjustable SE block.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super(ResNet3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
        self.se = SEBlock3D(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    """
    ResNet3D with adjustable parameters: number of layers, use of SE blocks, dropout.
    """
    def __init__(self, num_demographics, num_blocks_per_layer=[2, 2, 2], initial_filters=32, filters_multiplier=2, use_se=False, use_dropout=True, use_demographics=False):
        super(ResNet3D, self).__init__()
        self.num_blocks_per_layer = num_blocks_per_layer
        self.initial_filters = initial_filters
        self.filters_multiplier = filters_multiplier
        self.use_se = use_se
        self.use_dropout = use_dropout
        self.use_demographics = use_demographics

        self.conv1 = nn.Conv3d(1, initial_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        current_filters = initial_filters
        self.layer1 = self._make_layer(current_filters, current_filters, num_blocks_per_layer[0], use_se=use_se)
        self.layer2 = self._make_layer(current_filters, int(current_filters * filters_multiplier), num_blocks_per_layer[1], stride=2, use_se=use_se)
        current_filters = int(current_filters * filters_multiplier)
        self.layer3 = self._make_layer(current_filters, int(current_filters * filters_multiplier), num_blocks_per_layer[2], stride=2, use_se=use_se)
        current_filters = int(current_filters * filters_multiplier)



        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(current_filters, 64)
        self.dropout_layer = nn.Dropout(0.5) if use_dropout else nn.Identity()
        fc2_input_size = 64 + num_demographics if self.use_demographics else 64
        self.fc2 = nn.Linear(fc2_input_size, 1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1, use_se=False):
        layers = []
        layers.append(ResNet3DBlock(in_channels, out_channels, stride, use_se=use_se))
        for _ in range(1, num_blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x, demographics):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        if self.use_demographics:
            x = torch.cat((x, demographics), dim=1)
        x = self.fc2(x)
        return x

    def get_name(self):
        """Dynamically generate model name based on parameters."""
        name = "ResNet3D"
        name += f"_layers{'-'.join(map(str, self.num_blocks_per_layer))}_filters{self.initial_filters}_filtermult{self.filters_multiplier}"
        if self.use_se:
            name += "_SE"
        if self.use_dropout:
            name += "_DO"
        if self.use_demographics:
            name += "_with_demographics"
        else:
            name += "_without_demographics"
        return name

    def get_params(self):
        """Return model parameters for wandb config."""
        return {
            "num_blocks_per_layer": self.num_blocks_per_layer,
            "initial_filters": self.initial_filters,
            "filters_multiplier": self.filters_multiplier,
            "use_se": self.use_se,
            "use_dropout": self.use_dropout,
            "use_demographics": self.use_demographics,
            "architecture": "ResNet3D"
        }

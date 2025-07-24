import torch
import torch.nn as nn
from .resnext3d import SEBlock3D

class DenseBlock3D(nn.Module):
    """
    DenseBlock3D with adjustable SE block.
    """
    def __init__(self, in_channels, growth_rate, num_layers, use_se=False):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm3d(current_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(current_channels, growth_rate, kernel_size=3, padding=1),
            )
            if use_se:
                layer = nn.Sequential(layer, SEBlock3D(growth_rate))
            self.layers.append(layer)
            current_channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet3D(nn.Module):
    """
    DenseNet3D with adjustable parameters: growth rate, number of layers in DenseBlocks, use of SE blocks, dropout.
    """
    def __init__(self, num_demographics, growth_rate=16, num_dense_layers=[4, 4], initial_filters=32, transition_filters_multiplier=2, use_se_blocks=False, use_dropout=True, use_demographics=False):
        super(DenseNet3D, self).__init__()
        self.growth_rate = growth_rate
        self.num_dense_layers = num_dense_layers
        self.initial_filters = initial_filters
        self.transition_filters_multiplier = transition_filters_multiplier
        self.use_se_blocks = use_se_blocks
        self.use_dropout = use_dropout
        self.use_demographics = use_demographics

        self.conv1 = nn.Conv3d(1, initial_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        current_filters = initial_filters
        self.dense1 = DenseBlock3D(current_filters, growth_rate, num_dense_layers[0], use_se=use_se_blocks)
        current_filters += num_dense_layers[0] * growth_rate
        next_filters = int(current_filters * transition_filters_multiplier)
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(current_filters),
            nn.Conv3d(current_filters, next_filters, kernel_size=1),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )
        current_filters = next_filters

        self.dense2 = DenseBlock3D(current_filters, growth_rate, num_dense_layers[1], use_se=use_se_blocks)
        current_filters += num_dense_layers[1] * growth_rate
        next_filters = int(current_filters * transition_filters_multiplier)
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(current_filters),
            nn.Conv3d(current_filters, next_filters, kernel_size=1),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )
        current_filters = next_filters


        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(current_filters, 64)
        self.dropout_layer = nn.Dropout(0.5) if use_dropout else nn.Identity()
        fc2_input_size = 64 + num_demographics if self.use_demographics else 64
        self.fc2 = nn.Linear(fc2_input_size, 1)

    def forward(self, x, demographics):
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
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
        name = "DenseNet3D"
        name += f"_layers{'-'.join(map(str, self.num_dense_layers))}_gr{self.growth_rate}_filters{self.initial_filters}_transmult{self.transition_filters_multiplier}"
        if self.use_se_blocks:
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
            "growth_rate": self.growth_rate,
            "num_dense_layers": self.num_dense_layers,
            "initial_filters": self.initial_filters,
            "transition_filters_multiplier": self.transition_filters_multiplier,
            "use_se_blocks": self.use_se_blocks,
            "use_dropout": self.use_dropout,
            "use_demographics": self.use_demographics,
            "architecture": "DenseNet3D"
        }
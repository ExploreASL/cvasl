import torch
import torch.nn as nn
from .resnext3d import SEBlock3D

class Large3DCNN(nn.Module):
    """
        Large 3D CNN with adjustable parameters: number of convolutional layers, initial filters, filter multiplier, use of BN, dropout, SE blocks.
    """
    def __init__(self, num_demographics, num_conv_layers=3, initial_filters=16, filters_multiplier=2, use_bn=True, use_dropout=True, use_se=False, dropout_rate=0.2, use_demographics=False):
        super(Large3DCNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.initial_filters = initial_filters
        self.filters_multiplier = filters_multiplier
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.use_se = use_se
        self.dropout_rate = dropout_rate
        self.use_demographics = use_demographics
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        filters = initial_filters
        in_channels = 1

        for _ in range(self.num_conv_layers):
            if filters <= 0:
                filters = 1
            if in_channels <= 0:
                in_channels = 1

            conv_layer = nn.Conv3d(in_channels, filters, kernel_size=3, padding=1)
            self.conv_layers.append(conv_layer)
            if use_bn:
                self.bn_layers.append(nn.BatchNorm3d(filters))
            self.relu_layers.append(nn.ReLU())
            self.pool_layers.append(nn.MaxPool3d(kernel_size=2))
            if use_se:
                self.se_blocks.append(SEBlock3D(filters))
            in_channels = filters
            filters = int(filters * filters_multiplier)

        self.flatten = nn.Flatten()

        size = [120, 144, 120]
        for _ in range(num_conv_layers):
            size = [s // 2 for s in size]
        flattened_size = in_channels * size[0] * size[1] * size[2]


        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu4 = nn.ReLU()
        fc2_input_size = 128 + num_demographics if self.use_demographics else 128
        self.fc2 = nn.Linear(fc2_input_size, 1)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
    
    def forward(self, x, demographics):
        """Forward pass."""
        for i in range(len(self.conv_layers)):
            conv = self.conv_layers[i]
            x = conv(x)
            if self.use_bn and i < len(self.bn_layers):
                bn = self.bn_layers[i]
                x = bn(x)
            relu = self.relu_layers[i]
            x = relu(x)
            if self.use_se and i < len(self.se_blocks):
                se_block = self.se_blocks[i]
                x = se_block(x)
            pool = self.pool_layers[i]
            x = pool(x)

        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        if self.use_demographics:
            x = torch.cat((x, demographics), dim=1)
        x = self.fc2(x)
        return x

    def get_name(self):
        """Dynamically generate model name based on parameters."""
        name = "Large3DCNN"
        name += f"_layers{self.num_conv_layers}_filters{self.initial_filters}_filtermult{self.filters_multiplier}"
        if self.use_bn:
            name += "_BN"
        if self.use_se:
            name += "_SE"
        if self.use_dropout:
            name += "_DO"
        name += f"_dropout{self.dropout_rate}"
        if self.use_demographics:
            name += "_with_demographics"
        else:
            name += "_without_demographics"        
        return name

    def get_params(self):
        """Return model parameters for wandb config."""
        return {
            "num_conv_layers": self.num_conv_layers,
            "initial_filters": self.initial_filters,
            "filters_multiplier": self.filters_multiplier,
            "use_bn": self.use_bn,
            "use_se": self.use_se,
            "use_dropout": self.use_dropout,
            "dropout_rate": self.dropout_rate,
            "use_demographics": self.use_demographics,
            "architecture": "Large3DCNN"
        }

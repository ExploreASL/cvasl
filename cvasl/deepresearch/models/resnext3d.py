import torch
import torch.nn as nn

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation Block for 3D CNNs."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResNeXt3DBlock(nn.Module):
    """ResNeXt 3D Block."""
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, bottleneck_width=4, use_se=False):
        super().__init__()
        groups = cardinality
        width = max(1, int(out_channels * (bottleneck_width / 64.))) * groups
        inter_channels = groups * width

        self.conv1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(inter_channels, inter_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm3d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(inter_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock3D(out_channels) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += identity
        out = nn.ReLU(inplace=True)(out)
        return out

class ResNeXt3D(nn.Module):
    """ResNeXt3D Network."""
    def __init__(self, num_demographics, num_blocks_per_layer=[2, 2, 2], cardinality=32, bottleneck_width=4, initial_filters=64, filters_multiplier=2, use_se=False, use_dropout=True, use_demographics=False):
        super().__init__()
        self.num_blocks_per_layer = num_blocks_per_layer
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.initial_filters = initial_filters
        self.filters_multiplier = filters_multiplier
        self.use_se = use_se
        self.use_dropout = use_dropout
        self.use_demographics = use_demographics

        self.initial_filters = initial_filters
        self.conv1 = nn.Conv3d(1, self.initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        current_filters = self.initial_filters
        self.layer1 = self._make_layer(current_filters,   int(current_filters * filters_multiplier),   num_blocks_per_layer[0], stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width, use_se=use_se)
        current_filters = int(current_filters * filters_multiplier)
        self.layer2 = self._make_layer(current_filters,   int(current_filters * filters_multiplier),  num_blocks_per_layer[1], stride=2, cardinality=cardinality, bottleneck_width=bottleneck_width, use_se=use_se)
        current_filters = int(current_filters * filters_multiplier)
        self.layer3 = self._make_layer(current_filters,  int(current_filters * filters_multiplier), num_blocks_per_layer[2], stride=2, cardinality=cardinality, bottleneck_width=bottleneck_width, use_se=use_se)
        current_filters = int(current_filters * filters_multiplier)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(current_filters, 64)
        self.dropout_layer = nn.Dropout(0.5) if use_dropout else nn.Identity()
        fc2_input_size = 64 + num_demographics if self.use_demographics else 64
        self.fc2 = nn.Linear(fc2_input_size, 1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1, cardinality=32, bottleneck_width=4, use_se=False):
        layers = []
        layers.append(ResNeXt3DBlock(in_channels, out_channels, stride, cardinality, bottleneck_width, use_se))
        for _ in range(1, num_blocks):
            layers.append(ResNeXt3DBlock(out_channels, out_channels, 1, cardinality, bottleneck_width, use_se))
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
        name = "ResNeXt3D"
        name += f"_layers{'-'.join(map(str, self.num_blocks_per_layer))}_cardinality{self.cardinality}_bw{self.bottleneck_width}_filters{self.initial_filters}_filtermult{self.filters_multiplier}"
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
            "cardinality": self.cardinality,
            "bottleneck_width": self.bottleneck_width,
            "initial_filters": self.initial_filters,
            "filters_multiplier": self.filters_multiplier,
            "use_se": self.use_se,
            "use_dropout": self.use_dropout,
            "use_demographics": self.use_demographics,
            "architecture": "ResNeXt3D"
        }

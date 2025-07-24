import torch
import torch.nn as nn
from .resnext3d import SEBlock3D

class PreActResBlock3D(nn.Module):
    """Pre-activation ResNet3D Block with Bottleneck and SE."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        bottleneck_channels = out_channels // 4  # bottleneck channels
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm3d(bottleneck_channels)
        self.conv2 = nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm3d(bottleneck_channels)
        self.conv3 = nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1, bias=False)

        self.se = SEBlock3D(out_channels)  # add squeeze-and-excitation

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )


    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.se(out)

        out += self.shortcut(identity)
        return out



class Improved3DCNN(nn.Module):
    def __init__(self, num_demographics, initial_filters=32, filters_multiplier=2, num_conv_layers=3, use_se=True, dropout_rate=0.3, use_demographics=False):
        super().__init__()
        self.initial_filters = initial_filters
        self.filters_multiplier = filters_multiplier
        self.num_conv_layers = num_conv_layers
        self.use_se = use_se
        self.dropout_rate = dropout_rate
        self.use_demographics = use_demographics

        current_filters = initial_filters
        self.conv1 = nn.Conv3d(1, current_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(current_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(2)

        # dynamically create convolutional layers
        conv_layers = []
        for i in range(num_conv_layers -1): # -1 because first conv is outside
            in_channels = current_filters
            current_filters = int(current_filters * filters_multiplier)
            out_channels = current_filters
                         
            conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            conv_layers.append(conv_layer)
            conv_layers.append(nn.BatchNorm3d(out_channels))
            conv_layers.append(SEBlock3D(out_channels) if use_se else nn.Identity())
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool3d(2))
            

        self.conv_layers = nn.Sequential(*conv_layers)


        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()

        # correct flattened_size calculation, now it's truly dynamic
        flattened_size = current_filters  # after all multiplications


        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        fc2_input_size = 128 + num_demographics if self.use_demographics else 128
        self.fc2 = nn.Linear(fc2_input_size, 1)

            
    def forward(self, x, demographics):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.conv_layers(x)  # apply the dynamic conv layers
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        if self.use_demographics:
            x = torch.cat((x, demographics), dim=1)
        x = self.fc2(x)
        return x

    def get_name(self):
        name = f"Improved3DCNN_layers{self.num_conv_layers}_filters{self.initial_filters}_filtermult{self.filters_multiplier}"
        if self.use_se:
            name += "_SE"
        name += f"_dropout{self.dropout_rate}"
        if self.use_demographics:
            name += "_with_demographics"  # indicate when using demographics
        else:
            name += "_without_demographics"
        return name

    def get_params(self):
        return {
            "initial_filters": self.initial_filters,
            "filters_multiplier": self.filters_multiplier,
            "num_conv_layers": self.num_conv_layers,
            "use_se": self.use_se,
            "dropout_rate": self.dropout_rate,
            "use_demographics": self.use_demographics,
            "architecture": "Improved3DCNN",
        }


import torch
import torch.nn as nn

class VascularAttention3D(nn.Module):
    """
    Vascular Attention Module for 3D CNNs.
    Learns spatial attention weights to focus on vascular regions.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv3d(in_channels, 1, kernel_size=1) # 1x1 conv to learn attention map
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Vascular Attention Module.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Attention-weighted feature map of shape (B, C, D, H, W).
        """
        attention_map = self.sigmoid(self.attention_conv(x)) # (B, 1, D, H, W) - spatial attention map
        return x * attention_map # Apply attention weights to input features
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DropPath(nn.Module):
    """DropPath implements the stochastic depth mechanism."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class PatchEmbed3D(nn.Module):
    """
    Splits a 3D image into non-overlapping patches and projects them into a given embedding dimension.
    """
    def __init__(self, img_size=(120, 144, 120), patch_size=(16, 16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # flattening is done in forward

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({D}*{H}*{W}) doesn't match expected ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)  # DEBUG: (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # DEBUG: (B, num_patches, embed_dim)
        return x

class HybridEmbed3D(nn.Module):
    """CNN stem for 3D inputs before transformer."""
    def __init__(self, in_chans=1, embed_dim=768, kernel_size=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim // 4, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim // 4),
            nn.GELU(),
            nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=1, padding=1),
        )
    
    def forward(self, x):
        return self.proj(x)

class Attention3D(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """Standard MLP block used in Vision Transformers."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block3D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer3D(nn.Module):
    """
    Vision Transformer for 3D inputs.
    
    Parameters:
      - num_demographics: Number of demographic features.
      - use_demographics: Whether to fuse demographics with image features.
      - early_demographics_fusion: If True and using CLS token, fuse demographics early.
      - use_cls_token: If True, use a dedicated [CLS] token (ignoring global pooling).
      - use_hybrid_embed: If True, use a CNN stem (HybridEmbed3D) instead of patch embeddings.
      - hybrid_kernel_size: Kernel size for the CNN stem (if used).
    """
    def __init__(self, 
                 num_demographics,
                 img_size=(120, 144, 120), 
                 patch_size=(16, 16, 16), 
                 in_chans=1, 
                 num_classes=1, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, 
                 global_pool=True,
                 use_demographics=False,
                 early_demographics_fusion=False,
                 use_cls_token=False,
                 use_hybrid_embed=False,
                 hybrid_kernel_size=3):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.use_demographics = use_demographics
        self.early_demographics_fusion = early_demographics_fusion
        self.use_cls_token = use_cls_token
        self.use_hybrid_embed = use_hybrid_embed
        self.embed_dim = embed_dim  # store embedding dimension
        self.num_demographics = num_demographics  # store for classifier reset
        
        # Embedding: either patch-based or hybrid (CNN stem)
        if self.use_hybrid_embed:
            self.embed = HybridEmbed3D(in_chans, embed_dim, kernel_size=hybrid_kernel_size)
            # With two conv layers at stride=2, the reduction factor is 4
            num_tokens = (img_size[0] // 4) * (img_size[1] // 4) * (img_size[2] // 4)
        else:
            self.embed = PatchEmbed3D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_tokens = self.embed.num_patches

        # Positional embeddings (and optional CLS token)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_shape = (1, num_tokens + 1, embed_dim)
        else:
            pos_embed_shape = (1, num_tokens, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(*pos_embed_shape))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks with drop path schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block3D(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Early fusion: If using CLS token and early fusion, project demographics and add to CLS token.
        if self.use_cls_token and self.early_demographics_fusion:
            self.demog_proj = nn.Linear(num_demographics, embed_dim)

        # Classification head
        self.fc_norm = norm_layer(embed_dim)
        self.fc_head = nn.Linear(embed_dim, 128)
        fc_in_size = 128
        # Late fusion: if not early fusion, concatenate demographics after fc_head activation
        if not (self.use_cls_token and self.early_demographics_fusion) and self.use_demographics:
            fc_in_size = 128 + num_demographics
        self.fc_out = nn.Linear(fc_in_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        return self.fc_head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.fc_head = nn.Linear(self.embed_dim, 128)
        fc_in_size = 128
        if not (self.use_cls_token and self.early_demographics_fusion) and self.use_demographics:
            fc_in_size = 128 + self.num_demographics
        self.fc_out = nn.Linear(fc_in_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, demographics=None):
        B = x.shape[0]
        x = self.embed(x)  # If using hybrid embed: (B, embed_dim, D', H', W')
        if self.use_hybrid_embed:
            x = x.flatten(2).transpose(1, 2)  # Flatten CNN output to tokens: (B, num_tokens, embed_dim)
        # else, PatchEmbed3D already returns (B, num_tokens, embed_dim)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            if self.early_demographics_fusion and (demographics is not None):
                cls_tokens = cls_tokens + self.demog_proj(demographics).unsqueeze(1)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, num_tokens+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x, demographics=None):
        """
        Parameters:
          x: 3D image tensor of shape (B, C, D, H, W)
          demographics: tensor of demographic features of shape (B, num_demographics)
                        (Optional if demographics are not used or early fusion is enabled.)
        """
        x = self.forward_features(x, demographics)

        if self.use_cls_token:
            # Use the CLS token output.
            x_cls = x[:, 0]
            feat = self.fc_norm(x_cls)
        elif self.global_pool:
            # Global pooling across tokens.
            x = x.transpose(1, 2)  # (B, embed_dim, num_tokens)
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, embed_dim)
            feat = self.fc_norm(x)
        else:
            # Fallback: take the first token.
            feat = self.fc_norm(x[:, 0])

        x = self.relu(self.fc_head(feat))
        x = self.dropout(x)

        # late fusion: concatenate demographics if configured and not fused early.
        if (not (self.use_cls_token and self.early_demographics_fusion)) and self.use_demographics and (demographics is not None):
            x = torch.cat((x, demographics), dim=1)
        x = self.fc_out(x)
        return x

    def get_name(self):
        """Generate a dynamic model name based on parameters."""
        name = "ViT3D"
        if self.use_hybrid_embed:
            name += "_hybrid"
        else:
            name += f"_patch{'-'.join(map(str, self.embed.patch_size))}"
        name += f"_embed{self.embed_dim}_depth{len(self.blocks)}_heads{self.blocks[0].attn.num_heads}"
        if self.use_cls_token:
            name += "_with_cls"
        else:
            name += "_with_globalpool" if self.global_pool else "_no_pool"
        if self.use_demographics:
            if self.use_cls_token and self.early_demographics_fusion:
                name += "_early_demog"
            else:
                name += "_late_demog"
        return name

    def get_params(self):
        """Return model parameters for logging/configuration."""
        return {
            "patch_size": self.embed.patch_size if not self.use_hybrid_embed else "hybrid",
            "embed_dim": self.embed_dim,
            "depth": len(self.blocks),
            "num_heads": self.blocks[0].attn.num_heads,
            "use_demographics": self.use_demographics,
            "early_demographics_fusion": self.early_demographics_fusion,
            "use_cls_token": self.use_cls_token,
            "use_hybrid_embed": self.use_hybrid_embed,
            "architecture": "ViT3D"
        }

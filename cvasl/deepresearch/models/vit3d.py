import torch
import torch.nn as nn

class PatchEmbed3D(nn.Module):
    """ Converts a 3D image/volume into patches and then into embeddings
    """
    def __init__(self, img_size=(120, 144, 120), patch_size=(16, 16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2) 

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x) # B, embed_dim, num_patches_D, num_patches_H, num_patches_W
        x = x.flatten(2).transpose(1, 2) # B, num_patches, embed_dim
        return x

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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy and get on with your life, i.e. don't use tensor as tuple

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
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
        # NOTE: drop_path for stochastic depth, we shall see if this is useful or not mwahahaha
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer3D(nn.Module):
    """ Vision Transformer for 3D inputs (volumes)
    """
    def __init__(self, num_demographics, img_size=(120, 144, 120), patch_size=(16, 16, 16), in_chans=1, num_classes=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, global_pool=True, use_demographics=False):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.use_demographics = use_demographics

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate] * depth  # stochastic depth decay rule, double checked with the original implementation, is correct
        self.blocks = nn.Sequential(*[
            Block3D(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc_norm = norm_layer(embed_dim) if global_pool else nn.Identity()
        self.global_pool_layer = nn.AdaptiveAvgPool1d(1) if global_pool else nn.Identity()
        self.fc_head = nn.Linear(embed_dim, 128)
        fc_out_size = 128 + num_demographics if self.use_demographics else 128
        self.fc_out = nn.Linear(fc_out_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()


        # Initialize weights (optional, gives option for fine tuning, could be used for some sort of harmonization, another potential branch for further research)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
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
        self.fc_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, x, demographics):
        x = self.forward_features(x)

        if self.global_pool:
            x = self.global_pool_layer(x.transpose(1, 2)).transpose(1, 2) # (B, N, C) -> (B, C, 1) -> (B, 1, C)
            x = x[:, 0] # (B, 1, C) -> (B, C) - take the first and only token representation
        else:
            x = x[:, 0] # just take [CLS] token if exists - not using CLS token here in this implementation, might need to get back to this

        x = self.fc_norm(x)
        x = self.relu(self.fc_head(x))
        x = self.dropout(x)

        if self.use_demographics:
            x = torch.cat((x, demographics), dim=1)
        x = self.fc_out(x)
        return x

    def get_name(self):
        """Dynamically generate model name based on parameters."""
        name = "ViT3D"
        name += f"_patch{'-'.join(map(str, self.patch_embed.patch_size))}_embed{self.patch_embed.proj.out_channels}_depth{len(self.blocks)}_heads{self.blocks[0].attn.num_heads}"
        if self.use_demographics:
            name += "_with_demographics"
        else:
            name += "_without_demographics"
        return name

    def get_params(self):
        """Return model parameters for wandb config."""
        return {
            "patch_size": self.patch_embed.patch_size,
            "embed_dim": self.patch_embed.proj.out_channels,
            "depth": len(self.blocks),
            "num_heads": self.blocks[0].attn.num_heads,
            "use_demographics": self.use_demographics,
            "architecture": "ViT3D"
        }


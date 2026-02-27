"""
SAMUS ViT-B Image Encoder.
"""

import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d, MLPBlock


class FeatureAdapter(nn.Module):
    """Lightweight adapter injected into each transformer block.

    Structure: Linear(dim → dim//4) → ReLU → Linear(dim//4 → dim)
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = dim // reduction
        self.down = nn.Linear(dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class PositionAdapter(nn.Module):
    """Adapt SAM's 1024-resolution positional embedding to 256-resolution.

    Max-pool (stride=4) + 3×3 convolution for spatial adaptation.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_embed: [1, H, W, C] original positional embedding.
        Returns:
            [1, H', W', C] spatially adapted embedding.
        """
        x = pos_embed.permute(0, 3, 1, 2)     # [1, C, H, W]
        x = self.pool(x)
        x = self.conv(x)
        return x.permute(0, 2, 3, 1)          # [1, H', W', C]


class Attention(nn.Module):
    """Multi-head attention with relative position bias (SAM-style)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Single transformer block with optional feature adapter."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_adapter: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.adapter = FeatureAdapter(dim) if use_adapter else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.adapter(x)
        return x


class SAMUSImageEncoder(nn.Module):
    """SAMUS ViT-B image encoder (256×256 input, 16×16 output tokens).

    Used as F_pre for computing variance-spectrum anchor embeddings
    in the STRIQ anchor selection pipeline.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        use_adapters: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
        )
        self.pos_adapter = PositionAdapter(embed_dim) if use_adapters else None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_adapter=use_adapters,
            )
            for _ in range(depth)
        ])
        self.neck_norm = nn.LayerNorm(embed_dim)

        # Output neck: project to out_chans
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 256, 256] input ultrasound image.
        Returns:
            [B, 256, 16, 16] feature embedding.
        """
        # Patch embedding → [B, embed_dim, H', W']
        x = self.patch_embed(x)
        H, W = x.shape[2], x.shape[3]

        # Add positional embedding
        pos = self.pos_embed
        if self.pos_adapter is not None:
            pos = self.pos_adapter(pos)
        # Reshape for sequence processing: [B, N, C]
        x = x.permute(0, 2, 3, 1)          # [B, H', W', C]
        x = x + pos
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [B, N, C]

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.neck_norm(x)

        # Reshape back to spatial
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Output projection
        return self.neck(x)

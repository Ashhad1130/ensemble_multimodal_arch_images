import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=Config.EMBED_DIM, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(self.embed_dim, embed_dim * 3)
        self.proj = nn.Linear(self.embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x, attn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=Config.EMBED_DIM, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, return_attention=False):
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.attn(normed_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=Config.IMAGE_SIZE, patch_size=Config.PATCH_SIZE, in_channels=3, embed_dim=Config.EMBED_DIM):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return self.dropout(x)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=Config.IMAGE_SIZE, patch_size=Config.PATCH_SIZE, embed_dim=Config.EMBED_DIM, depth=6, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, return_attention=False):
        x = self.patch_embed(x)
        attention_weights = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_weights.append(attn)
            else:
                x = block(x)
        x = self.norm(x)
        cls_features = x[:, 0]
        if return_attention:
            return cls_features, attention_weights
        return cls_features
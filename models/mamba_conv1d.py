import torch
import torch.nn as nn
from config import Config

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

class MambaConv1DBlock(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=4):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba is not available. Please install mamba-ssm")
        
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            x_residual = x
            x = layer_norm(x)
            x = mamba_layer(x)
            x = x + x_residual
            x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x = self.output_proj(x)
        return x + residual
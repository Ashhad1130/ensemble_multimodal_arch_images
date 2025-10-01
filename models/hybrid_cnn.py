import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class MultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.small_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.medium_branch = nn.Sequential(
            nn.Conv2d(3, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.large_branch = nn.Sequential(
            nn.Conv2d(3, 256, 7, 4, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 7, 1, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 128 + 256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        small_feat = self.small_branch(x)
        medium_feat = self.medium_branch(x)
        large_feat = self.large_branch(x)
        
        target_size = 8
        small_feat = F.interpolate(small_feat, size=target_size, mode='bilinear', align_corners=False)
        medium_feat = F.interpolate(medium_feat, size=target_size, mode='bilinear', align_corners=False)
        
        fused_feat = torch.cat([small_feat, medium_feat, large_feat], dim=1)
        fused_feat = self.fusion(fused_feat)
        fused_feat = fused_feat.flatten(1)
        
        attention_weights = self.attention(fused_feat)
        attended_feat = fused_feat * attention_weights
        
        return attended_feat
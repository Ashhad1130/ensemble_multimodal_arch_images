import torch
import torch.nn as nn
from config import Config

class EnhancedGenerator(nn.Module):
    """Enhanced Generator for AnoGAN"""
    
    def __init__(self, noise_dim=Config.NOISE_DIM):
        super().__init__()
        self.noise_dim = noise_dim
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, noise):
        return self.main(noise)

class EnhancedDiscriminator(nn.Module):
    """Enhanced Discriminator for AnoGAN"""
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_features=False):
        features = self.features(x)
        output = self.classifier(features)
        
        if return_features:
            return output, features.view(features.size(0), -1)
        return output
import torch
import torch.nn as nn
from config import Config

class EnhancedAutoencoder(nn.Module):
    """Enhanced autoencoder with attention mechanisms"""
    
    def __init__(self, latent_dim=Config.LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.enc_fc = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 14 * 14),
            nn.BatchNorm1d(512 * 14 * 14),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.enc_fc(features)
    
    def decode(self, z):
        features = self.dec_fc(z)
        features = features.view(-1, 512, 14, 14)
        return self.decoder(features)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
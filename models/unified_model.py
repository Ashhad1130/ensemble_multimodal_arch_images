import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from models.autoencoder import EnhancedAutoencoder
from models.anogan import EnhancedGenerator, EnhancedDiscriminator
from models.vit import VisionTransformer
from models.hybrid_cnn import MultiScaleCNN
from models.mamba_conv1d import MambaConv1DBlock, MAMBA_AVAILABLE
from models.backbone import get_backbone_model
import logging

logger = logging.getLogger(__name__)

class MultimodalAnoGAN(nn.Module):
    """Independent multimodal model: Autoencoder + AnoGAN + Backbone"""
    
    def __init__(self, backbone_name=Config.BACKBONE):
        super().__init__()
        self.architecture = "multimodal_anogan"
        self.backbone_name = backbone_name
        
        # Initialize components
        self.autoencoder = EnhancedAutoencoder(Config.LATENT_DIM)
        self.generator = EnhancedGenerator(Config.NOISE_DIM)
        self.discriminator = EnhancedDiscriminator()
        self.backbone, self.backbone_dim = get_backbone_model(backbone_name)
        
        # Feature fusion and classifier
        self._create_classifier()
        logger.info(f"Initialized Multimodal AnoGAN with {backbone_name}")
    
    def _create_classifier(self):
        # Combine features from all components
        total_dim = self.backbone_dim + Config.LATENT_DIM + 512
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, Config.NUM_DISEASES)
        )
    
    def generate_samples(self, batch_size, device):
        """Generate fake samples for GAN training"""
        noise = torch.randn(batch_size, self.generator.noise_dim, 1, 1, device=device)
        return self.generator(noise)
    
    def forward(self, x):
        # Backbone features
        backbone_features = self.backbone(x)
        backbone_features = backbone_features.view(backbone_features.size(0), -1)
        
        # Autoencoder features
        reconstruction, ae_features = self.autoencoder(x)
        
        # Discriminator features (real images only)
        _, disc_features = self.discriminator(x, return_features=True)
        
        # Combine all features
        combined_features = torch.cat([backbone_features, ae_features, disc_features], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return {
            'logits': logits,
            'features': combined_features,
            'reconstruction': reconstruction,
            'architecture': self.architecture
        }

class MultimodalViT(nn.Module):
    """Independent multimodal model: Autoencoder + ViT + Backbone"""
    
    def __init__(self, backbone_name=Config.BACKBONE):
        super().__init__()
        self.architecture = "multimodal_vit"
        self.backbone_name = backbone_name
        
        # Initialize components
        self.autoencoder = EnhancedAutoencoder(Config.LATENT_DIM)
        self.vit = VisionTransformer(
            img_size=Config.IMAGE_SIZE,
            patch_size=Config.PATCH_SIZE,
            embed_dim=Config.EMBED_DIM,
            depth=6,
            num_heads=12
        )
        self.backbone, self.backbone_dim = get_backbone_model(backbone_name)
        
        # Feature fusion and classifier
        self._create_classifier()
        logger.info(f"Initialized Multimodal ViT with {backbone_name}")
    
    def _create_classifier(self):
        # Combine features from all components
        total_dim = self.backbone_dim + Config.LATENT_DIM + Config.EMBED_DIM
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, Config.NUM_DISEASES)
        )
    
    def forward(self, x, return_attention=False):
        # Backbone features
        backbone_features = self.backbone(x)
        backbone_features = backbone_features.view(backbone_features.size(0), -1)
        
        # Autoencoder features
        reconstruction, ae_features = self.autoencoder(x)
        
        # ViT features
        if return_attention:
            vit_features, attention_weights = self.vit(x, return_attention=True)
        else:
            vit_features = self.vit(x)
        
        # Combine all features
        combined_features = torch.cat([backbone_features, ae_features, vit_features], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        outputs = {
            'logits': logits,
            'features': combined_features,
            'reconstruction': reconstruction,
            'architecture': self.architecture
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs

class MultimodalHybrid(nn.Module):
    """Independent multimodal model: Autoencoder + Hybrid CNN + Backbone"""
    
    def __init__(self, backbone_name=Config.BACKBONE):
        super().__init__()
        self.architecture = "multimodal_hybrid"
        self.backbone_name = backbone_name
        
        # Initialize components
        self.autoencoder = EnhancedAutoencoder(Config.LATENT_DIM)
        self.hybrid_cnn = MultiScaleCNN()
        self.backbone, self.backbone_dim = get_backbone_model(backbone_name)
        
        # Feature fusion and classifier
        self._create_classifier()
        logger.info(f"Initialized Multimodal Hybrid with {backbone_name}")
    
    def _create_classifier(self):
        # Combine features from all components
        total_dim = self.backbone_dim + Config.LATENT_DIM + 512  # Hybrid outputs 512
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, Config.NUM_DISEASES)
        )
    
    def forward(self, x):
        # Backbone features
        backbone_features = self.backbone(x)
        backbone_features = backbone_features.view(backbone_features.size(0), -1)
        
        # Autoencoder features
        reconstruction, ae_features = self.autoencoder(x)
        
        # Hybrid CNN features
        hybrid_features = self.hybrid_cnn(x)
        
        # Combine all features
        combined_features = torch.cat([backbone_features, ae_features, hybrid_features], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return {
            'logits': logits,
            'features': combined_features,
            'reconstruction': reconstruction,
            'architecture': self.architecture
        }

class MultimodalMamba(nn.Module):
    """Independent multimodal model: Mamba + Conv1D + Backbone"""
    
    def __init__(self, backbone_name=Config.BACKBONE):
        super().__init__()
        self.architecture = "multimodal_mamba"
        self.backbone_name = backbone_name
        
        # Initialize components
        self.backbone, self.backbone_dim = get_backbone_model(backbone_name)
        
        # Compute dynamic dimensions for Mamba
        self.seq_len = Config.MAMBA_SEQ_LEN
        self.feat_per_token = self.backbone_dim // self.seq_len  # Assume divides evenly; padding handles remainder
        self.hidden_dim = self.feat_per_token // 2 if self.feat_per_token // 2 > 0 else 128  # Reasonable default
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is required for MultimodalMamba")
        
        self.mamba_conv1d = MambaConv1DBlock(
            input_dim=self.feat_per_token,
            hidden_dim=self.hidden_dim,
            num_layers=4
        )
        
        # Feature fusion and classifier (no autoencoder, so no reconstruction)
        self._create_classifier()
        logger.info(f"Initialized Multimodal Mamba with {backbone_name}")
    
    def _create_classifier(self):
        total_dim = self.feat_per_token  # After mean(dim=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, Config.NUM_DISEASES)
        )
    
    def forward(self, x):
        # Backbone features
        backbone_features = self.backbone(x)
        backbone_features = backbone_features.view(backbone_features.size(0), -1)
        
        # Mamba processing
        batch_size, feat_dim = backbone_features.shape
        seq_len = self.seq_len
        
        if feat_dim % seq_len != 0:
            pad_size = seq_len - (feat_dim % seq_len)
            backbone_features = F.pad(backbone_features, (0, pad_size))
            feat_dim = backbone_features.shape[1]
        
        feat_per_token = feat_dim // seq_len
        mamba_input = backbone_features.view(batch_size, feat_per_token, seq_len)
        mamba_output = self.mamba_conv1d(mamba_input)
        final_features = mamba_output.mean(dim=2)
        
        # Classification
        logits = self.classifier(final_features)
        
        return {
            'logits': logits,
            'features': final_features,
            'architecture': self.architecture
        }

class EnsembleAll(nn.Module):
    """Ensemble of all 4 multimodal architectures"""
    
    def __init__(self, backbone_name=Config.BACKBONE):
        super().__init__()
        self.architecture = "ensemble_all"
        
        # Initialize all independent models
        self.multimodal_anogan = MultimodalAnoGAN(backbone_name)
        self.multimodal_vit = MultimodalViT(backbone_name)
        self.multimodal_hybrid = MultimodalHybrid(backbone_name)
        self.multimodal_mamba = MultimodalMamba(backbone_name)
        
        # Compute dimensions dynamically
        self.backbone_dim = self.multimodal_anogan.backbone_dim
        anogan_dim = self.backbone_dim + Config.LATENT_DIM + 512
        vit_dim = self.backbone_dim + Config.LATENT_DIM + Config.EMBED_DIM
        hybrid_dim = self.backbone_dim + Config.LATENT_DIM + 512
        mamba_dim = self.multimodal_mamba.feat_per_token
        
        # Common projection dimension
        self.common_feature_dim = 512
        
        # Projection layers to align features
        self.proj_anogan = nn.Linear(anogan_dim, self.common_feature_dim)
        self.proj_vit = nn.Linear(vit_dim, self.common_feature_dim)
        self.proj_hybrid = nn.Linear(hybrid_dim, self.common_feature_dim)
        self.proj_mamba = nn.Linear(mamba_dim, self.common_feature_dim)
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(4, dtype=torch.float32) / 4)
        
        logger.info("Initialized Ensemble of all 4 multimodal architectures")
    
    def forward(self, x):
        # Get predictions from all models
        outputs_anogan = self.multimodal_anogan(x)
        outputs_vit = self.multimodal_vit(x)
        outputs_hybrid = self.multimodal_hybrid(x)
        outputs_mamba = self.multimodal_mamba(x)
        
        all_logits = [
            outputs_anogan['logits'],
            outputs_vit['logits'], 
            outputs_hybrid['logits'],
            outputs_mamba['logits']
        ]
        
        # Project features to common dim
        projected_features = [
            self.proj_anogan(outputs_anogan['features']),
            self.proj_vit(outputs_vit['features']),
            self.proj_hybrid(outputs_hybrid['features']),
            self.proj_mamba(outputs_mamba['features'])
        ]
        
        # Apply softmax to ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted average of logits
        ensemble_logits = torch.zeros_like(all_logits[0])
        for i, logits in enumerate(all_logits):
            ensemble_logits += weights[i] * logits
        
        # Average projected features
        ensemble_features = torch.mean(torch.stack(projected_features), dim=0)
        
        return {
            'logits': ensemble_logits,
            'features': ensemble_features,
            'architecture': self.architecture,
            'individual_logits': all_logits,
            'ensemble_weights': weights
        }

class UnifiedChestXRayModel(nn.Module):
    """Unified model that selects the desired architecture"""
    
    def __init__(self, architecture=Config.ARCHITECTURE, backbone_name=Config.BACKBONE):
        super().__init__()
        self.architecture = architecture
        self.backbone_name = backbone_name
        
        # Select the appropriate model
        if architecture == "multimodal_anogan":
            self.model = MultimodalAnoGAN(backbone_name)
        elif architecture == "multimodal_vit":
            self.model = MultimodalViT(backbone_name)
        elif architecture == "multimodal_hybrid":
            self.model = MultimodalHybrid(backbone_name)
        elif architecture == "multimodal_mamba":
            self.model = MultimodalMamba(backbone_name)
        elif architecture == "ensemble_all":
            self.model = EnsembleAll(backbone_name)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        logger.info(f"Initialized {architecture} with {backbone_name}")
    
    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)
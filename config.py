import torch

class Config:
    """Configuration for the Chest X-Ray Analysis System"""
    
    # Data parameters
    DATA_ROOT = "chest_xray"
    IMAGE_DIR = "images"
    CSV_PATH = "sample_labels.csv"
    
    # CSV column mapping
    CSV_COLUMNS = {
        'image_index': 'Image Index',
        'finding_labels': 'Finding Labels'
    }
    
    # Available architectures (removed ensemble_all)
    MULTIMODAL_ARCHITECTURES = {
        'multimodal_anogan': {
            'name': 'Autoencoder + AnoGAN + Backbone',
            'components': ['autoencoder', 'anogan', 'backbone'],
            'enabled': True
        },
        'multimodal_vit': {
            'name': 'Autoencoder + ViT + Backbone', 
            'components': ['autoencoder', 'vit', 'backbone'],
            'enabled': True
        },
        'multimodal_hybrid': {
            'name': 'Autoencoder + Hybrid CNN + Backbone',
            'components': ['autoencoder', 'hybrid_cnn', 'backbone'],
            'enabled': True
        },
        'multimodal_mamba': {
            'name': 'Mamba + Conv1D + Backbone',
            'components': ['mamba_conv1d', 'backbone'],
            'enabled': True
        }
    }
    
    # Model parameters
    ARCHITECTURE = "multimodal_hybrid"  # Options: multimodal_anogan, multimodal_vit, multimodal_hybrid, multimodal_mamba
    BACKBONE = "resnet50"  # resnet50 or densenet121
    OUTPUT_DIR = ARCHITECTURE + '_' + BACKBONE
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LATENT_DIM = 512
    NOISE_DIM = 100
    EMBED_DIM = 768
    PATCH_SIZE = 16
    
    # Mamba parameters
    MAMBA_PATCH_SIZE = 14  # 224/16 = 14 patches per dimension
    MAMBA_SEQ_LEN = MAMBA_PATCH_SIZE * MAMBA_PATCH_SIZE  # 196 patches total
    MAMBA_HIDDEN_DIM = 256
    
    # Training parameters
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 1
    EARLY_STOPPING_PATIENCE = 5
    
    # Loss weights (configurable)
    LOSS_WEIGHTS = {
        'classification': 1.0,
        'reconstruction': 0.1,
        'adversarial': 0.01
    }
    
    # Feature fusion parameters
    ATTENTION_HEADS = 8
    FUSION_DROPOUT = 0.1
    
    # Device detection with MPS support
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        MIXED_PRECISION = True
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        MIXED_PRECISION = False
    else:
        DEVICE = torch.device("cpu")
        MIXED_PRECISION = False
    
    # Evaluation parameters
    BOOTSTRAP_SAMPLES = 1000
    CONFIDENCE_LEVEL = 0.95
    STATISTICAL_SIGNIFICANCE_ALPHA = 0.05
    
    # Disease classes
    PATHOLOGY_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    NUM_DISEASES = len(PATHOLOGY_NAMES)
    
    # Multi-label stratification parameters
    MIN_SAMPLES_PER_CLASS = 10
    STRATIFY_THRESHOLD = 0.05  # Minimum class frequency for stratification
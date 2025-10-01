# Chest X-Ray Analysis System

A comprehensive PyTorch-based system for multi-label classification of thoracic pathologies in chest X-ray images. This project implements advanced multimodal architectures combining autoencoders, generative models, transformers, hybrid CNNs, and state-space models (Mamba) with robust backbones like ResNet50 or DenseNet121.

Designed for the NIH ChestX-ray14 dataset, it supports stratified data splitting, class imbalance handling, mixed-precision training, and extensive evaluation with statistical significance testing.

## Features

- **Multi-Label Classification**: Detects 14 common thoracic pathologies (e.g., Atelectasis, Cardiomegaly, Pneumonia)
- **Multimodal Architectures**:
  - Autoencoder + AnoGAN + Backbone
  - Autoencoder + Vision Transformer (ViT) + Backbone
  - Autoencoder + Hybrid Multi-Scale CNN + Backbone
  - Mamba (State-Space Model) + Conv1D + Backbone
- **Ensemble Support**: Weighted fusion of all architectures
- **Advanced Training**:
  - Mixed-precision training (CUDA/MPS/CPU)
  - GAN training for AnoGAN variant
  - Class-weighted BCE loss with inverse frequency weighting
  - Early stopping, LR scheduling, and checkpointing
- **Data Handling**:
  - Multi-label stratified splitting for imbalanced datasets
  - Robust image loading with quality checks and error handling
  - Data augmentation for training
- **Evaluation**:
  - ROC-AUC, PR-AUC, F1-score with per-disease breakdowns
  - Bootstrap confidence intervals and t-tests for significance
  - Calibration curves, ROC/PR plots, and confusion matrices
- **Logging & Visualization**: Comprehensive logging, performance plots, and summary reports

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/chest-xray-analysis.git
cd chest-xray-analysis
2. Create a Virtual Environment (recommended)
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bashpip install torch torchvision torchaudio  # Adjust for your CUDA version if needed
pip install pandas scikit-learn pillow tqdm numpy matplotlib seaborn scipy
For Mamba support (optional, required for multimodal_mamba):
bashpip install mamba-ssm causal-conv1d>=1.2.0
Full requirements.txt:
texttorch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
scikit-learn>=1.2.0
Pillow>=9.0.0
tqdm>=4.65.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
mamba-ssm>=1.0.0  # Optional
causal-conv1d>=1.2.0  # For Mamba
4. Verify Installation
bashpython -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
Dataset
It is present in the root directory


The system automatically handles label parsing (comma-separated pathologies)
Note: For testing, use the provided sample_labels.csv 
Configuration
Edit config.py to customize:

Architecture: "multimodal_hybrid" (default). Options: multimodal_anogan, multimodal_vit, multimodal_hybrid, multimodal_mamba
Backbone: "resnet50" or "densenet121"
Data: DATA_ROOT = "chest_xray", IMAGE_SIZE = 224, BATCH_SIZE = 32
Training: LEARNING_RATE = 1e-4, EPOCHS = 1 (increase for full training), LOSS_WEIGHTS
Device: Auto-detects CUDA/MPS/CPU

Pathologies (14 classes)
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, 
Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, 
Pleural_Thickening, Hernia
Usage
Quick Start
Run the full pipeline (data prep → training → evaluation):
bashpython main.py
Outputs: Models in {architecture}_{backbone}/, logs, plots, CSVs.
Example: For ViT + DenseNet, set ARCHITECTURE = "multimodal_vit", BACKBONE = "densenet121".
Training Only
pythonfrom pipeline import UnifiedPipeline
from config import Config

Config.ARCHITECTURE = "multimodal_hybrid"  # Customize
Config.EPOCHS = 50

pipeline = UnifiedPipeline()
history = pipeline.train_model()  # Returns training history
Evaluation Only
Load a trained model and evaluate:
pythonfrom evaluator import ComprehensiveEvaluator
from torch.utils.data import DataLoader
from dataset import ChestXRayDataset
import torch

# Load model
model = torch.load("multimodal_hybrid_resnet50/best_model_multimodal_hybrid.pth")['model_state_dict']
model = UnifiedChestXRayModel.load_state_dict(model)
model.eval()

# Prepare test loader
test_dataset = ChestXRayDataset(...)  # See dataset.py
test_loader = DataLoader(test_dataset, batch_size=32)

# Evaluate
evaluator = ComprehensiveEvaluator(model)
metrics, results = evaluator.evaluate(test_loader)
print(f"Mean AUC: {metrics['mean_auc']:.4f}")
Custom Runs

Change Splits: Modify train_size, val_size in pipeline.py
Ensemble: Set ARCHITECTURE = "ensemble_all" (computational heavy)
Debug: Set EPOCHS = 1, reduce BATCH_SIZE for low-memory setups

Models
ArchitectureComponentsStrengthsmultimodal_anoganAutoencoder + AnoGAN + BackboneAnomaly detection, reconstructionmultimodal_vitAutoencoder + ViT + BackboneGlobal attention, long-range depsmultimodal_hybridAutoencoder + Multi-Scale CNN + BackboneMulti-resolution feature fusionmultimodal_mambaMamba + Conv1D + BackboneEfficient sequence modelingensemble_allAll above (weighted)Best overall performance
All models output logits for 14 pathologies and support feature extraction.
Results
Expected Metrics (on ChestX-ray14 test set, after 50 epochs):

Mean AUC: ~0.85–0.92 (varies by architecture)
Top Performers: Cardiomegaly (~0.95 AUC), Effusion (~0.93)
Statistical Tests: Typically p < 0.001 vs. random (AUC=0.5)

Generated Artifacts

best_model_{arch}.pth: Best checkpoint
predictions.csv, labels.csv: Test predictions
Plots: ROC/PR curves, calibration, confusion matrices (*_{arch}.png)
evaluation_summary.txt: Human-readable report
performance_summary_{arch}.csv: Metrics table

Troubleshooting

OOM Errors: Reduce BATCH_SIZE or use MIXED_PRECISION = False
No CUDA: Falls back to CPU/MPS; training slower
Mamba ImportError: Install mamba-ssm or avoid multimodal_mamba
Data Issues: Ensure CSV has no NaNs; check logs for skipped samples
Reproducibility: Fixed seeds (42); use torch.backends.cudnn.deterministic = True

Contributing

Fork the repo
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add some AmazingFeature')
Push to branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Built on PyTorch and torchvision
Inspired by NIH ChestX-ray14 and state-of-the-art medical imaging papers
Thanks to the Mamba-SSM team for efficient sequence modeling
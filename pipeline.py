import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import os
import json
import time
import logging
from config import Config
from dataset import ChestXRayDataset
from models.unified_model import UnifiedChestXRayModel
from trainer import AdvancedTrainer
from evaluator import ComprehensiveEvaluator
import pandas as pd

logger = logging.getLogger(__name__)

class UnifiedPipeline:
    def __init__(self):
        self.model = None
        self.trainer = None
        self.evaluator = None
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # Set deterministic algorithms for reproducibility
        if Config.DEVICE.type == 'cuda':
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        logger.info("Pipeline initialized with fixed random seeds")
    
    def prepare_data_loaders(self):
        """Prepare data loaders with multi-label stratification"""
        logger.info("Preparing data loaders with multi-label stratification...")
        
        try:
            # Create transforms
            train_transforms = self._get_train_transforms()
            val_test_transforms = self._get_val_test_transforms()
            
            # Load full dataset
            full_dataset = ChestXRayDataset(
                data_dir=Config.DATA_ROOT,
                csv_path=Config.CSV_PATH,
                image_dir=Config.IMAGE_DIR,
                transform=None,  # Will be set later
                mode='full',
                quality_check=True
            )
            
            if len(full_dataset) == 0:
                raise ValueError("No valid samples found in dataset")
            
            # Get multi-label stratified split indices
            train_indices, val_indices, test_indices = full_dataset.get_multilabel_split_indices(
                train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
            )
            
            # Create subset datasets
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)
            
            # Set transforms for each subset
            train_dataset.dataset.transform = train_transforms
            val_dataset.dataset.transform = val_test_transforms
            test_dataset.dataset.transform = val_test_transforms
            
            # Create data loaders with appropriate settings
            pin_memory = Config.DEVICE.type == 'cuda'
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=pin_memory,
                drop_last=True,
                persistent_workers=Config.NUM_WORKERS > 0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=pin_memory,
                persistent_workers=Config.NUM_WORKERS > 0
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=pin_memory,
                persistent_workers=Config.NUM_WORKERS > 0
            )
            
            logger.info(f"Data loaders created successfully:")
            logger.info(f"  Training: {len(train_dataset)} samples")
            logger.info(f"  Validation: {len(val_dataset)} samples")
            logger.info(f"  Testing: {len(test_dataset)} samples")
            logger.info(f"  Batch size: {Config.BATCH_SIZE}, Workers: {Config.NUM_WORKERS}")
            
            return train_loader, val_loader, test_loader, full_dataset.class_weights
            
        except Exception as e:
            logger.error(f"Failed to prepare data loaders: {e}")
            raise
    
    def _get_train_transforms(self):
        """Get training transforms with augmentation"""
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
            transforms.RandomCrop((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_test_transforms(self):
        """Get validation/test transforms without augmentation"""
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup_model(self):
        """Setup model with comprehensive error handling"""
        logger.info(f"Setting up model: {Config.ARCHITECTURE}")
        
        try:
            self.model = UnifiedChestXRayModel(Config.ARCHITECTURE, Config.BACKBONE)
            
            # Calculate model size and parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024**2)  # Assuming float32
            
            logger.info(f"Model created successfully:")
            logger.info(f"  Architecture: {Config.ARCHITECTURE}")
            logger.info(f"  Backbone: {Config.BACKBONE}")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Model size: ~{model_size_mb:.1f} MB")
            logger.info(f"  Device: {Config.DEVICE}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def train_model(self, train_loader, val_loader, class_weights, epochs):
        """Train model with comprehensive error handling"""
        logger.info("Starting model training...")
        
        try:
            self.trainer = AdvancedTrainer(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights
            )
            
            history = self.trainer.train(epochs)
            
            logger.info("Training completed successfully!")
            logger.info(f"Best validation AUC: {self.trainer.best_auc:.4f}")
            logger.info(f"Best epoch: {self.trainer.best_epoch + 1}")
            
            return history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, test_loader):
        """Evaluate model with comprehensive error handling"""
        logger.info("Starting comprehensive evaluation...")
        
        try:
            # Load best model if available
            checkpoint_path = f'{self.model.architecture}/best_model_{self.model.architecture}.pth'
            if os.path.exists(checkpoint_path):
                logger.info("Loading best model for evaluation...")
                try:
                    if Config.DEVICE.type == 'cuda':
                        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
                    else:
                        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded model from epoch {checkpoint['epoch']} with AUC {checkpoint['best_auc']:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load best model ({e}), using current model")
            else:
                logger.warning("No saved model found, using current model")
            
            # Create evaluator and run evaluation
            self.evaluator = ComprehensiveEvaluator(self.model)
            metrics, results = self.evaluator.evaluate(test_loader)
            
            logger.info("Evaluation completed successfully!")
            return metrics, results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def save_final_results(self, metrics, history):
        """Save comprehensive final results"""
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            results_file = f'{CONFIG.OUTPUT_DIR}/final_results_{Config.ARCHITECTURE}_{timestamp}.json'
            
            final_results = {
                'system_info': {
                    'architecture': Config.ARCHITECTURE,
                    'backbone': Config.BACKBONE,
                    'device': str(Config.DEVICE),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'pytorch_version': torch.__version__,
                    'config': {
                        'image_size': Config.IMAGE_SIZE,
                        'batch_size': Config.BATCH_SIZE,
                        'learning_rate': Config.LEARNING_RATE,
                        'weight_decay': Config.WEIGHT_DECAY,
                        'epochs': Config.EPOCHS,
                        'loss_weights': Config.LOSS_WEIGHTS,
                        'mixed_precision': Config.MIXED_PRECISION
                    }
                },
                'evaluation_metrics': metrics,
                'training_history': history,
                'model_info': {
                    'total_parameters': sum(p.numel() for p in self.model.parameters()),
                    'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                },
                'performance_summary': {
                    'mean_auc': metrics.get('mean_auc', 0.0),
                    'auc_95_ci': [metrics.get('auc_95_ci_lower', 0.0), metrics.get('auc_95_ci_upper', 0.0)],
                    'significantly_better_than_random': metrics.get('significantly_better_than_random', False),
                    'best_diseases': self._get_top_diseases(metrics, 'auc', top_k=5),
                    'worst_diseases': self._get_bottom_diseases(metrics, 'auc', bottom_k=5)
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info(f"Final results saved: {results_file}")
            
            # Create summary table
            self._create_performance_summary_table(metrics)
            
            return results_file
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
            return None
    
    def _get_top_diseases(self, metrics, metric_name, top_k=5):
        """Get top performing diseases"""
        disease_scores = []
        for disease in Config.PATHOLOGY_NAMES:
            score = metrics.get(f'{disease}_{metric_name}', 0.0)
            disease_scores.append((disease, score))
        
        disease_scores.sort(key=lambda x: x[1], reverse=True)
        return disease_scores[:top_k]
    
    def _get_bottom_diseases(self, metrics, metric_name, bottom_k=5):
        """Get bottom performing diseases"""
        disease_scores = []
        for disease in Config.PATHOLOGY_NAMES:
            score = metrics.get(f'{disease}_{metric_name}', 0.0)
            disease_scores.append((disease, score))
        
        disease_scores.sort(key=lambda x: x[1])
        return disease_scores[:bottom_k]
    
    def _create_performance_summary_table(self, metrics):
        """Create and save performance summary table"""
        try:
            # Create performance summary
            performance_data = []
            for disease in Config.PATHOLOGY_NAMES:
                performance_data.append({
                    'Disease': disease,
                    'AUC': f"{metrics.get(f'{disease}_auc', 0.0):.4f}",
                    'PR-AUC': f"{metrics.get(f'{disease}_pr_auc', 0.0):.4f}",
                    'F1': f"{metrics.get(f'{disease}_f1_score', 0.0):.4f}",
                    'Prevalence': f"{metrics.get(f'{disease}_prevalence', 0.0):.4f}",
                    'Positive_Samples': metrics.get(f'{disease}_positive_samples', 0),
                    'Optimal_Threshold': f"{metrics.get(f'{disease}_optimal_threshold', 0.5):.4f}"
                })
            
            df = pd.DataFrame(performance_data)
            df.to_csv(f'{CONFIG.OUTPUT_DIR}/performance_summary_{Config.ARCHITECTURE}.csv', index=False)
            logger.info(f"Performance summary saved: {CONFIG.OUTPUT_DIR}/performance_summary_{Config.ARCHITECTURE}.csv")

        except Exception as e:
            logger.warning(f"Failed to create performance summary table: {e}")
    
    def run_pipeline(self):
        """Run complete pipeline with comprehensive error handling"""
        try:
            logger.info("="*60)
            logger.info(f"STARTING CHEST X-RAY ANALYSIS PIPELINE")
            logger.info(f"Architecture: {Config.ARCHITECTURE}")
            logger.info(f"Device: {Config.DEVICE}")
            logger.info("="*60)
            
            # Validate data setup
            if not self._validate_data_setup():
                raise ValueError("Data validation failed")
            
            # Prepare data loaders
            train_loader, val_loader, test_loader, class_weights = self.prepare_data_loaders()
            
            # Setup model
            model = self.setup_model()
            
            # Train model
            history = self.train_model(train_loader, val_loader, class_weights, Config.EPOCHS)
            
            # Evaluate model
            metrics, results = self.evaluate_model(test_loader)
            
            # Save final results
            results_file = self.save_final_results(metrics, history)
            
            # Print final summary
            self._print_final_summary(metrics)
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            
            return metrics, results, history
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error("="*60)
            logger.error("PIPELINE FAILED!")
            logger.error("="*60)
            raise
    
    def _print_final_summary(self, metrics):
        """Print final performance summary"""
        logger.info("\nFINAL PERFORMANCE SUMMARY:")
        logger.info("-" * 40)
        logger.info(f"Mean AUC: {metrics.get('mean_auc', 0):.4f} ± {metrics.get('std_auc', 0):.4f}")
        logger.info(f"Mean PR-AUC: {metrics.get('mean_pr_auc', 0):.4f}")
        logger.info(f"Mean F1: {metrics.get('mean_f1', 0):.4f}")
        
        if 'auc_95_ci_lower' in metrics:
            logger.info(f"95% CI: [{metrics['auc_95_ci_lower']:.4f}, {metrics['auc_95_ci_upper']:.4f}]")
        
        significance = metrics.get('significantly_better_than_random', False)
        logger.info(f"Significantly better than random: {'Yes' if significance else 'No'}")
        
        # Top and bottom performers
        top_diseases = self._get_top_diseases(metrics, 'auc', 3)
        bottom_diseases = self._get_bottom_diseases(metrics, 'auc', 3)
        
        logger.info(f"\nTop 3 diseases by AUC:")
        for disease, auc in top_diseases:
            logger.info(f"  {disease}: {auc:.4f}")
        
        logger.info(f"\nBottom 3 diseases by AUC:")
        for disease, auc in bottom_diseases:
            logger.info(f"  {disease}: {auc:.4f}")
    
    def _validate_data_setup(self):
        """Validate data setup with comprehensive checks"""
        issues = []
        
        try:
            # Check data directory
            if not os.path.exists(Config.DATA_ROOT):
                issues.append(f"Data directory not found: {Config.DATA_ROOT}")
            
            # Check CSV file
            csv_path = os.path.join(Config.DATA_ROOT, Config.CSV_PATH)
            if not os.path.exists(csv_path):
                issues.append(f"CSV file not found: {csv_path}")
            else:
                try:
                    df = pd.read_csv(csv_path)
                    required_cols = [Config.CSV_COLUMNS['image_index'], Config.CSV_COLUMNS['finding_labels']]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        issues.append(f"CSV missing required columns: {missing_cols}")
                    else:
                        logger.info(f"CSV validation passed: {len(df)} rows found")
                        
                        # Check for data quality
                        empty_images = df[Config.CSV_COLUMNS['image_index']].isna().sum()
                        empty_labels = df[Config.CSV_COLUMNS['finding_labels']].isna().sum()
                        if empty_images > 0:
                            logger.warning(f"Found {empty_images} rows with missing image names")
                        if empty_labels > 0:
                            logger.warning(f"Found {empty_labels} rows with missing labels")
                            
                except Exception as e:
                    issues.append(f"Error reading CSV file: {e}")
            
            # Check image directory
            image_dir = os.path.join(Config.DATA_ROOT, Config.IMAGE_DIR)
            if not os.path.exists(image_dir):
                issues.append(f"Image directory not found: {image_dir}")
            else:
                image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
                image_files = [f for f in os.listdir(image_dir) 
                              if any(f.lower().endswith(ext) for ext in image_extensions)]
                logger.info(f"Image directory validation passed: {len(image_files)} image files found")
                
                if len(image_files) == 0:
                    issues.append("No image files found in image directory")
                elif len(image_files) < 100:
                    logger.warning(f"Very few images found ({len(image_files)}). This may not be sufficient for training.")
            
            # Check device availability
            if Config.DEVICE.type == 'cuda' and not torch.cuda.is_available():
                issues.append("CUDA device specified but not available")
            
            # Report results
            if issues:
                logger.error("\nData Setup Issues:")
                for issue in issues:
                    logger.error(f"  • {issue}")
                return False
            else:
                logger.info("\nData validation passed successfully!")
                return True
                
        except Exception as e:
            logger.error(f"Data validation failed with error: {e}")
            return False
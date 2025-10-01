import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import numpy as np
from config import Config

logger = logging.getLogger(__name__)

class AdvancedTrainer:
    def __init__(self, model, train_loader, val_loader, class_weights=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.architecture = model.architecture
        
        # Setup optimizers
        self.optimizer = self._setup_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Setup loss functions
        if class_weights is not None:
            class_weights = class_weights.to(Config.DEVICE)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        self.mse_criterion = nn.MSELoss()
        
        # GAN setup only for AnoGAN architecture
        self._setup_gan_training()
        
        # Mixed precision setup
        self._setup_mixed_precision()
        
        self.best_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_history = []
        
        logger.info(f"Trainer initialized for {self.architecture}")
    
    def _setup_optimizer(self):
        """Setup optimizer with proper parameter grouping"""
        return optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
    
    def _setup_gan_training(self):
        """Setup GAN training components if needed"""
        if self.architecture == "multimodal_anogan":
            self.bce_gan = nn.BCELoss()
            self.gen_optimizer = optim.AdamW(
                self.model.model.generator.parameters(), 
                lr=Config.LEARNING_RATE * 0.5,  # Lower LR for generator
                weight_decay=Config.WEIGHT_DECAY
            )
            self.disc_optimizer = optim.AdamW(
                self.model.model.discriminator.parameters(), 
                lr=Config.LEARNING_RATE * 0.5,  # Lower LR for discriminator
                weight_decay=Config.WEIGHT_DECAY
            )
            self.enable_gan = True
            logger.info("GAN training enabled for AnoGAN architecture")
        else:
            self.enable_gan = False
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        if Config.MIXED_PRECISION and Config.DEVICE.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast = torch.cuda.amp.autocast
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            self.autocast = self._dummy_autocast
    
    def _dummy_autocast(self, enabled=True):
        """Dummy autocast for non-CUDA devices"""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()
    
    def _calculate_losses(self, outputs, images, labels):
        """Calculate all loss components - return both tensors and float values"""
        
        # Classification loss
        cls_loss = self.criterion(outputs['logits'], labels)
        
        # Reconstruction loss (if available)
        recon_loss = torch.tensor(0.0, device=Config.DEVICE)
        if 'reconstruction' in outputs and outputs['reconstruction'] is not None:
            recon_loss = self.mse_criterion(outputs['reconstruction'], images)
        
        # Total loss with configurable weights
        total_loss = (Config.LOSS_WEIGHTS['classification'] * cls_loss + 
                     Config.LOSS_WEIGHTS['reconstruction'] * recon_loss)
        
        # Return both tensors for backprop and float values for logging
        return {
            'tensors': {
                'total': total_loss,
                'classification': cls_loss,
                'reconstruction': recon_loss
            },
            'values': {
                'total': total_loss.item(),
                'classification': cls_loss.item(),
                'reconstruction': recon_loss.item(),
                'adversarial': 0.0  # Will be updated in GAN step
            }
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch with improved error handling"""
        self.model.train()
        
        # Loss tracking
        losses = {
            'total': 0.0,
            'classification': 0.0, 
            'reconstruction': 0.0,
            'adversarial': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move data to device
                images = batch['image'].to(Config.DEVICE, non_blocking=True)
                labels = batch['labels'].to(Config.DEVICE, non_blocking=True)
                
                # Skip if batch is too small
                if images.size(0) < 2:
                    continue
                
                # Reset gradients
                self.optimizer.zero_grad()
                if self.enable_gan:
                    self.gen_optimizer.zero_grad()
                    self.disc_optimizer.zero_grad()
                
                # Forward pass with autocast
                with self.autocast():
                    outputs = self.model(images)
                    loss_results = self._calculate_losses(outputs, images, labels)
                
                # Backward pass for main model
                total_loss_tensor = loss_results['tensors']['total']
                if self.scaler:
                    self.scaler.scale(total_loss_tensor).backward(retain_graph=self.enable_gan)
                    self.scaler.step(self.optimizer)
                else:
                    total_loss_tensor.backward(retain_graph=self.enable_gan)
                    self.optimizer.step()
                
                # GAN training step
                if self.enable_gan:
                    gan_loss = self._train_gan_step(images, outputs)
                    loss_results['values']['adversarial'] = gan_loss
                
                # Update scaler
                if self.scaler:
                    self.scaler.update()
                
                # Accumulate losses for logging
                for key, value in loss_results['values'].items():
                    losses[key] += value
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss_results["values"]["total"]:.4f}',
                    'Cls': f'{loss_results["values"]["classification"]:.4f}',
                    'Rec': f'{loss_results["values"]["reconstruction"]:.4f}',
                    'Adv': f'{loss_results["values"]["adversarial"]:.4f}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"CUDA OOM at batch {batch_idx}. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"Training error at batch {batch_idx}: {e}")
                    raise
            except Exception as e:
                logger.warning(f"Training batch {batch_idx} failed: {e}")
                continue
        
        if num_batches == 0:
            logger.warning("No successful training batches in epoch")
            return {key: 0.0 for key in losses.keys()}
        
        # Average losses
        return {key: value / num_batches for key, value in losses.items()}
    
    def _train_gan_step(self, real_images, outputs):
        """Improved GAN training step"""
        batch_size = real_images.size(0)
        device = real_images.device
        
        try:
            # Labels for GAN training
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # === Train Discriminator ===
            self.disc_optimizer.zero_grad()
            
            # Real images
            with self.autocast():
                real_pred, _ = self.model.model.discriminator(real_images, return_features=True)
                d_real_loss = self.bce_gan(real_pred, real_labels)
            
            # Generated images
            with torch.no_grad():
                fake_images = self.model.model.generate_samples(batch_size, device)
            
            with self.autocast():
                fake_pred, _ = self.model.model.discriminator(fake_images, return_features=True)
                d_fake_loss = self.bce_gan(fake_pred, fake_labels)
                
                d_loss = (d_real_loss + d_fake_loss) / 2
            
            # Backward pass for discriminator
            if self.scaler:
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.disc_optimizer)
            else:
                d_loss.backward()
                self.disc_optimizer.step()
            
            # === Train Generator ===
            self.gen_optimizer.zero_grad()
            
            # Generate new fake images for generator training
            fake_images_gen = self.model.model.generate_samples(batch_size, device)
            
            with self.autocast():
                fake_pred_gen, _ = self.model.model.discriminator(fake_images_gen, return_features=True)
                g_loss = self.bce_gan(fake_pred_gen, real_labels)  # Fool discriminator
            
            # Backward pass for generator
            if self.scaler:
                self.scaler.scale(g_loss).backward()
                self.scaler.step(self.gen_optimizer)
            else:
                g_loss.backward()
                self.gen_optimizer.step()
            
            return (d_loss.item() + g_loss.item()) * Config.LOSS_WEIGHTS['adversarial']
            
        except Exception as e:
            logger.warning(f"GAN training step failed: {e}")
            return 0.0
    
    def validate(self, epoch):
        """Validation with comprehensive error handling"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f'Validating Epoch {epoch+1}')):
                try:
                    images = batch['image'].to(Config.DEVICE, non_blocking=True)
                    labels = batch['labels'].to(Config.DEVICE, non_blocking=True)
                    
                    # Skip small batches
                    if images.size(0) < 1:
                        continue
                    
                    outputs = self.model(images)
                    loss_results = self._calculate_losses(outputs, images, labels)
                    
                    predictions = torch.sigmoid(outputs['logits'])
                    all_predictions.append(predictions.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    
                    total_loss += loss_results['values']['total']
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"CUDA OOM during validation at batch {batch_idx}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
                except Exception as e:
                    logger.warning(f"Validation batch {batch_idx} failed: {e}")
                    continue
        
        if num_batches == 0 or not all_predictions:
            logger.warning("No successful validation batches")
            return 0.0, {'val_loss': float('inf'), 'mean_auc': 0.0}
        
        try:
            predictions = np.concatenate(all_predictions, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            metrics = self._calculate_metrics(predictions, labels)
            metrics['val_loss'] = total_loss / num_batches
            return metrics['mean_auc'], metrics
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return 0.0, {'val_loss': total_loss / num_batches, 'mean_auc': 0.0}
    
    def _calculate_metrics(self, predictions, labels):
        """Calculate validation metrics with error handling"""
        from sklearn.metrics import roc_auc_score
        
        metrics = {}
        disease_aucs = []
        
        for i, disease in enumerate(Config.PATHOLOGY_NAMES):
            try:
                if len(np.unique(labels[:, i])) > 1:
                    auc_score = roc_auc_score(labels[:, i], predictions[:, i])
                    disease_aucs.append(auc_score)
                    metrics[f'{disease}_auc'] = auc_score
                else:
                    disease_aucs.append(0.5)  # Random performance
                    metrics[f'{disease}_auc'] = 0.5
                    
            except Exception as e:
                logger.warning(f"AUC calculation failed for {disease}: {e}")
                disease_aucs.append(0.5)
                metrics[f'{disease}_auc'] = 0.5
        
        if disease_aucs:
            metrics['mean_auc'] = np.mean(disease_aucs)
            metrics['std_auc'] = np.std(disease_aucs)
            metrics['min_auc'] = np.min(disease_aucs)
            metrics['max_auc'] = np.max(disease_aucs)
            metrics['median_auc'] = np.median(disease_aucs)
        else:
            metrics.update({
                'mean_auc': 0.0,
                'std_auc': 0.0,
                'min_auc': 0.0,
                'max_auc': 0.0,
                'median_auc': 0.0
            })
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_auc': self.best_auc,
                'architecture': self.architecture,
                'metrics': metrics,
                'config': {
                    'learning_rate': Config.LEARNING_RATE,
                    'batch_size': Config.BATCH_SIZE,
                    'architecture': Config.ARCHITECTURE
                }
            }
            
            # Add GAN optimizers if enabled
            if self.enable_gan:
                checkpoint['gen_optimizer_state_dict'] = self.gen_optimizer.state_dict()
                checkpoint['disc_optimizer_state_dict'] = self.disc_optimizer.state_dict()
            
            if is_best:
                torch.save(checkpoint, f'{CONFIG.OUTPUT_DIR}/best_model_{self.architecture}.pth', _use_new_zipfile_serialization=False)
                logger.info(f"Best model checkpoint saved: AUC={metrics.get('mean_auc', 0):.4f}")
            else:
                torch.save(checkpoint, f'{CONFIG.OUTPUT_DIR}/checkpoint_epoch_{epoch}_{self.architecture}.pth', _use_new_zipfile_serialization=False)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def train(self, epochs):
        """Main training loop with comprehensive error handling"""
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Architecture: {self.architecture}")
        logger.info(f"Device: {Config.DEVICE}")
        logger.info(f"Mixed Precision: {Config.MIXED_PRECISION}")
        logger.info(f"GAN Training: {self.enable_gan}")
        
        try:
            for epoch in range(epochs):
                # Training phase
                train_losses = self.train_epoch(epoch)
                
                # Validation phase
                val_auc, val_metrics = self.validate(epoch)
                
                # Learning rate scheduling
                self.scheduler.step(val_auc)
                
                # Create epoch history
                epoch_history = {
                    'epoch': epoch + 1,
                    'train_losses': train_losses,
                    'val_auc': val_auc,
                    'val_metrics': val_metrics,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                self.train_history.append(epoch_history)
                
                # Logging
                logger.info(f"Epoch {epoch+1}/{epochs}")
                logger.info(f"  Train Losses - Total: {train_losses['total']:.4f}, "
                           f"Cls: {train_losses['classification']:.4f}, "
                           f"Rec: {train_losses['reconstruction']:.4f}, "
                           f"Adv: {train_losses['adversarial']:.4f}")
                logger.info(f"  Validation AUC: {val_auc:.4f}")
                logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Save checkpoints
                if val_auc > self.best_auc:
                    self.best_auc = val_auc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    logger.info(f"  ✓ New best model! AUC: {self.best_auc:.4f}")
                else:
                    self.patience_counter += 1
                    
                # Regular checkpoint
                if epoch % 5 == 0:
                    self.save_checkpoint(epoch, val_metrics, is_best=False)
                
                # Early stopping
                if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
                # Memory cleanup
                if Config.DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
            
            logger.info(f"Training completed!")
            logger.info(f"Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch+1}")
            return self.train_history
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return self.train_history
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
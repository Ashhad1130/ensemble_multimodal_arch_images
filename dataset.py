import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import logging
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config

logger = logging.getLogger(__name__)

class MultilabelStratifiedSplitter:
    """Handles multi-label stratified splitting"""
    
    def __init__(self, min_samples_per_class=Config.MIN_SAMPLES_PER_CLASS):
        self.min_samples_per_class = min_samples_per_class
    
    def split(self, X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Multi-label stratified split
        Args:
            X: indices or features
            y: multi-label array [n_samples, n_classes]
            train_size, val_size, test_size: split proportions
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
        
        n_samples, n_classes = y.shape
        indices = np.arange(n_samples)
        
        # Check which classes have enough samples for stratification
        class_counts = y.sum(axis=0)
        stratifiable_classes = class_counts >= self.min_samples_per_class
        
        logger.info(f"Multi-label stratification: {stratifiable_classes.sum()}/{n_classes} classes have enough samples")
        
        if stratifiable_classes.sum() == 0:
            logger.warning("No classes have enough samples for stratification. Using random split.")
            return self._random_split(indices, train_size, val_size, test_size, random_state)
        
        # Use most frequent class combinations for stratification
        stratify_labels = self._get_stratification_labels(y, stratifiable_classes)
        
        try:
            # First split: train vs (val + test)
            train_indices, temp_indices = train_test_split(
                indices,
                train_size=train_size,
                stratify=stratify_labels,
                random_state=random_state
            )
            
            # Second split: val vs test
            val_prop = val_size / (val_size + test_size)
            temp_stratify = stratify_labels[temp_indices]
            
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_prop,
                stratify=temp_stratify,
                random_state=random_state
            )
            
            logger.info(f"Stratified split successful: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
            return train_indices, val_indices, test_indices
            
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}). Using random split.")
            return self._random_split(indices, train_size, val_size, test_size, random_state)
    
    def _get_stratification_labels(self, y, stratifiable_classes):
        """Create stratification labels from multi-label matrix"""
        # Use binary representation of class combinations
        stratify_labels = []
        for i in range(y.shape[0]):
            # Only consider stratifiable classes
            label_combo = tuple(y[i, stratifiable_classes].astype(int))
            stratify_labels.append(label_combo)
        
        # Convert to single labels for stratification
        unique_combos = list(set(stratify_labels))
        combo_to_label = {combo: idx for idx, combo in enumerate(unique_combos)}
        
        return np.array([combo_to_label[combo] for combo in stratify_labels])
    
    def _random_split(self, indices, train_size, val_size, test_size, random_state):
        """Fallback random split"""
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(indices)
        
        n_samples = len(indices)
        train_end = int(train_size * n_samples)
        val_end = train_end + int(val_size * n_samples)
        
        train_indices = shuffled_indices[:train_end]
        val_indices = shuffled_indices[train_end:val_end]
        test_indices = shuffled_indices[val_end:]
        
        return train_indices, val_indices, test_indices

class ChestXRayDataset(Dataset):
    """Enhanced dataset with improved quality checks and error handling"""
    
    def __init__(self, data_dir, csv_path, image_dir, transform=None, mode='train', quality_check=True):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, csv_path)
        self.image_dir = os.path.join(data_dir, image_dir)
        self.transform = transform
        self.mode = mode
        self.quality_check = quality_check
        self.pathology_names = Config.PATHOLOGY_NAMES
        
        try:
            self._validate_paths()
            self.dataframe = self._load_dataframe()
            self.samples = self._process_samples()
            self.class_weights = self._calculate_class_weights()
            
            logger.info(f"Dataset loaded: {len(self.samples)} samples for {mode}")
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {e}")
            raise
    
    def _validate_paths(self):
        """Validate all required paths exist"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
    
    def _load_dataframe(self):
        """Load and validate CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            required_columns = [Config.CSV_COLUMNS['image_index'], Config.CSV_COLUMNS['finding_labels']]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV missing required columns: {missing_cols}")
            logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def _process_samples(self):
        """Process samples with improved error handling"""
        samples = []
        skipped_count = 0
        error_reasons = {'missing_file': 0, 'corrupt_image': 0, 'size_too_small': 0}
        
        for idx, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe), desc="Processing samples"):
            try:
                image_name = str(row[Config.CSV_COLUMNS['image_index']]).strip()
                if not image_name or image_name == 'nan':
                    skipped_count += 1
                    error_reasons['missing_file'] += 1
                    continue
                
                image_path = os.path.join(self.image_dir, image_name)
                
                if not os.path.exists(image_path):
                    skipped_count += 1
                    error_reasons['missing_file'] += 1
                    continue
                
                if self.quality_check:
                    try:
                        with Image.open(image_path) as img:
                            if img.size[0] < 64 or img.size[1] < 64:
                                skipped_count += 1
                                error_reasons['size_too_small'] += 1
                                continue
                    except Exception:
                        skipped_count += 1
                        error_reasons['corrupt_image'] += 1
                        continue
                
                # Process labels
                finding_labels_str = str(row[Config.CSV_COLUMNS['finding_labels']])
                if finding_labels_str == 'nan' or not finding_labels_str:
                    finding_labels = ['No Finding']
                else:
                    finding_labels = finding_labels_str.split('|')
                
                label_vector = torch.zeros(len(self.pathology_names), dtype=torch.float32)
                
                has_disease = False
                for finding in finding_labels:
                    finding = finding.strip()
                    if finding == 'No Finding':
                        continue
                    if finding in self.pathology_names:
                        idx_disease = self.pathology_names.index(finding)
                        label_vector[idx_disease] = 1.0
                        has_disease = True
                
                samples.append({
                    'image_path': image_path,
                    'labels': label_vector,
                    'filename': image_name,
                    'has_disease': has_disease
                })
                
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                skipped_count += 1
                error_reasons['corrupt_image'] += 1
                continue
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} samples:")
            for reason, count in error_reasons.items():
                if count > 0:
                    logger.warning(f"  {reason}: {count}")
        
        if not samples:
            raise ValueError("No valid samples found in dataset")
        
        return samples
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        if not self.samples:
            return torch.ones(len(self.pathology_names))
        
        disease_counts = torch.zeros(len(self.pathology_names))
        total_samples = len(self.samples)
        
        for sample in self.samples:
            disease_counts += sample['labels']
        
        # Use inverse frequency with smoothing
        weights = torch.log(total_samples / (disease_counts + 1.0))
        weights = torch.clamp(weights, min=0.1, max=10.0)  # Prevent extreme weights
        
        # Normalize weights
        weights = weights / weights.mean()
        
        logger.info("Class weights calculated:")
        for i, (disease, weight) in enumerate(zip(self.pathology_names, weights)):
            pos_count = int(disease_counts[i])
            pos_ratio = pos_count / total_samples * 100
            logger.info(f"  {disease}: {pos_count} samples ({pos_ratio:.1f}%), weight: {weight:.3f}")
        
        return weights
    
    def get_multilabel_split_indices(self, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """Get indices for multi-label stratified split"""
        if not self.samples:
            raise ValueError("No samples available for splitting")
        
        # Create label matrix
        labels_matrix = torch.stack([sample['labels'] for sample in self.samples]).numpy()
        indices = np.arange(len(self.samples))
        
        # Use multi-label stratified splitter
        splitter = MultilabelStratifiedSplitter()
        train_indices, val_indices, test_indices = splitter.split(
            indices, labels_matrix, train_size, val_size, test_size, random_state
        )
        
        return train_indices, val_indices, test_indices
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                transform = transforms.Compose([
                    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = transform(image)
            
            return {
                'image': image,
                'labels': sample['labels'],
                'filename': sample['filename'],
                'has_disease': torch.tensor(float(sample['has_disease']))
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({sample['filename']}): {e}")
            # Return a dummy sample to prevent training crashes
            dummy_image = torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            dummy_labels = torch.zeros(len(self.pathology_names))
            return {
                'image': dummy_image,
                'labels': dummy_labels,
                'filename': f'dummy_{idx}',
                'has_disease': torch.tensor(0.0)
            }
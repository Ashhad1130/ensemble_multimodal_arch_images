import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc, 
    confusion_matrix, classification_report, f1_score
)
from scipy import stats
from tqdm import tqdm
import os
import json
import time
import pandas as pd
from config import Config
import logging

logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    def __init__(self, model):
        self.model = model.to(Config.DEVICE)
        self.model.eval()
        self.architecture = model.architecture
    
    def evaluate(self, test_loader, save_predictions=True):
        """Comprehensive evaluation with error handling"""
        logger.info("Starting comprehensive evaluation...")
        
        try:
            # Collect predictions and labels
            predictions, labels, filenames = self._collect_predictions(test_loader)
            
            if len(predictions) == 0:
                raise ValueError("No predictions collected during evaluation")
            
            logger.info(f"Evaluation completed on {len(predictions)} samples")
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(predictions, labels)
            
            # Create visualizations
            self._create_visualizations(predictions, labels, metrics)
            
            # Save results if requested
            if save_predictions:
                self._save_predictions(predictions, labels, filenames, metrics)
            
            return metrics, {
                'predictions': predictions,
                'labels': labels,
                'filenames': filenames
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _collect_predictions(self, test_loader):
        """Collect predictions with robust error handling"""
        all_predictions = []
        all_labels = []
        all_filenames = []
        failed_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                try:
                    images = batch['image'].to(Config.DEVICE, non_blocking=True)
                    labels = batch['labels'].to(Config.DEVICE, non_blocking=True)
                    
                    # Skip empty batches
                    if images.size(0) == 0:
                        continue
                    
                    outputs = self.model(images)
                    predictions = torch.sigmoid(outputs['logits'])
                    
                    all_predictions.append(predictions.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    all_filenames.extend(batch['filename'])
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"CUDA OOM during evaluation at batch {batch_idx}")
                        torch.cuda.empty_cache()
                        failed_batches += 1
                        continue
                    else:
                        raise
                except Exception as e:
                    logger.warning(f"Evaluation batch {batch_idx} failed: {e}")
                    failed_batches += 1
                    continue
        
        if failed_batches > 0:
            logger.warning(f"Failed to evaluate {failed_batches} batches")
        
        if not all_predictions:
            raise ValueError("No successful evaluation batches")
        
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return predictions, labels, all_filenames
    
    def _calculate_comprehensive_metrics(self, predictions, labels):
        """Calculate comprehensive metrics with statistical testing"""
        metrics = {
            'architecture': self.architecture,
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(predictions),
            'num_diseases': len(Config.PATHOLOGY_NAMES)
        }
        
        # Per-disease metrics
        disease_metrics = self._calculate_per_disease_metrics(predictions, labels)
        metrics.update(disease_metrics)
        
        # Overall statistics
        overall_stats = self._calculate_overall_statistics(disease_metrics)
        metrics.update(overall_stats)
        
        # Bootstrap confidence intervals
        bootstrap_results = self._bootstrap_confidence_intervals(predictions, labels)
        metrics.update(bootstrap_results)
        
        # Statistical significance testing
        significance_results = self._statistical_significance_tests(predictions, labels)
        metrics.update(significance_results)
        
        return metrics
    
    def _calculate_per_disease_metrics(self, predictions, labels):
        """Calculate detailed metrics for each disease"""
        metrics = {}
        
        logger.info("Per-disease AUC scores:")
        logger.info("-" * 60)
        
        for i, disease in enumerate(Config.PATHOLOGY_NAMES):
            disease_metrics = {}
            
            try:
                y_true = labels[:, i]
                y_pred = predictions[:, i]
                
                # Check if we have both classes
                if len(np.unique(y_true)) > 1:
                    # ROC-AUC
                    auc_score = roc_auc_score(y_true, y_pred)
                    disease_metrics['auc'] = auc_score
                    
                    # Precision-Recall AUC
                    precision, recall, _ = precision_recall_curve(y_true, y_pred)
                    pr_auc = auc(recall, precision)
                    disease_metrics['pr_auc'] = pr_auc
                    
                    # ROC curve data
                    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
                    disease_metrics['fpr'] = fpr.tolist()
                    disease_metrics['tpr'] = tpr.tolist()
                    disease_metrics['roc_thresholds'] = roc_thresholds.tolist()
                    
                    # Optimal threshold (Youden's J statistic)
                    optimal_idx = (tpr - fpr).argmax()
                    optimal_threshold = roc_thresholds[optimal_idx]
                    disease_metrics['optimal_threshold'] = optimal_threshold
                    disease_metrics['optimal_sensitivity'] = tpr[optimal_idx]
                    disease_metrics['optimal_specificity'] = 1 - fpr[optimal_idx]
                    
                    # F1 score at optimal threshold
                    y_pred_binary = (y_pred > optimal_threshold).astype(int)
                    f1 = f1_score(y_true, y_pred_binary)
                    disease_metrics['f1_score'] = f1
                    
                    # Sample counts
                    pos_samples = int(np.sum(y_true))
                    neg_samples = len(y_true) - pos_samples
                    disease_metrics['positive_samples'] = pos_samples
                    disease_metrics['negative_samples'] = neg_samples
                    disease_metrics['prevalence'] = pos_samples / len(y_true)
                    
                    logger.info(f"{disease:<20}: AUC={auc_score:.4f}, PR-AUC={pr_auc:.4f}, "
                               f"F1={f1:.4f}, Pos={pos_samples}, Prev={disease_metrics['prevalence']:.3f}")
                
                else:
                    # No positive samples or no negative samples
                    logger.warning(f"{disease:<20}: No variation in labels")
                    disease_metrics.update({
                        'auc': 0.5, 'pr_auc': 0.5, 'f1_score': 0.0,
                        'optimal_threshold': 0.5, 'optimal_sensitivity': 0.5,
                        'optimal_specificity': 0.5, 'positive_samples': int(np.sum(y_true)),
                        'negative_samples': int(len(y_true) - np.sum(y_true)),
                        'prevalence': np.mean(y_true)
                    })
                
            except Exception as e:
                logger.warning(f"Metrics calculation failed for {disease}: {e}")
                disease_metrics.update({
                    'auc': 0.5, 'pr_auc': 0.5, 'f1_score': 0.0,
                    'optimal_threshold': 0.5, 'optimal_sensitivity': 0.5,
                    'optimal_specificity': 0.5, 'positive_samples': 0,
                    'negative_samples': 0, 'prevalence': 0.0
                })
            
            # Add disease metrics to main metrics dict
            for metric_name, value in disease_metrics.items():
                metrics[f'{disease}_{metric_name}'] = value
        
        return metrics
    
    def _calculate_overall_statistics(self, disease_metrics):
        """Calculate overall statistics across all diseases"""
        # Extract AUC scores
        auc_scores = []
        pr_auc_scores = []
        f1_scores = []
        
        for disease in Config.PATHOLOGY_NAMES:
            auc_scores.append(disease_metrics[f'{disease}_auc'])
            pr_auc_scores.append(disease_metrics[f'{disease}_pr_auc'])
            f1_scores.append(disease_metrics[f'{disease}_f1_score'])
        
        stats = {
            'mean_auc': np.mean(auc_scores),
            'std_auc': np.std(auc_scores),
            'median_auc': np.median(auc_scores),
            'min_auc': np.min(auc_scores),
            'max_auc': np.max(auc_scores),
            'mean_pr_auc': np.mean(pr_auc_scores),
            'std_pr_auc': np.std(pr_auc_scores),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores)
        }
        
        logger.info("-" * 60)
        logger.info("Overall Statistics:")
        logger.info(f"  Mean AUC: {stats['mean_auc']:.4f} ± {stats['std_auc']:.4f}")
        logger.info(f"  AUC Range: [{stats['min_auc']:.4f}, {stats['max_auc']:.4f}]")
        logger.info(f"  Mean PR-AUC: {stats['mean_pr_auc']:.4f} ± {stats['std_pr_auc']:.4f}")
        logger.info(f"  Mean F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
        
        return stats
    
    def _bootstrap_confidence_intervals(self, predictions, labels):
        """Calculate bootstrap confidence intervals"""
        logger.info("Calculating bootstrap confidence intervals...")
        
        try:
            bootstrap_aucs = []
            n_samples = len(predictions)
            
            for _ in range(Config.BOOTSTRAP_SAMPLES):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                boot_pred = predictions[indices]
                boot_labels = labels[indices]
                
                # Calculate AUCs for this bootstrap sample
                boot_aucs = []
                for i in range(Config.NUM_DISEASES):
                    if len(np.unique(boot_labels[:, i])) > 1:
                        try:
                            auc = roc_auc_score(boot_labels[:, i], boot_pred[:, i])
                            boot_aucs.append(auc)
                        except:
                            pass
                
                if boot_aucs:
                    bootstrap_aucs.append(np.mean(boot_aucs))
            
            if bootstrap_aucs:
                ci_lower = np.percentile(bootstrap_aucs, (100 - Config.CONFIDENCE_LEVEL * 100) / 2)
                ci_upper = np.percentile(bootstrap_aucs, 100 - (100 - Config.CONFIDENCE_LEVEL * 100) / 2)
                
                results = {
                    'bootstrap_mean_auc': np.mean(bootstrap_aucs),
                    'bootstrap_std_auc': np.std(bootstrap_aucs),
                    'auc_95_ci_lower': ci_lower,
                    'auc_95_ci_upper': ci_upper,
                    'bootstrap_samples_used': len(bootstrap_aucs)
                }
                
                logger.info(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                return results
            else:
                logger.warning("Bootstrap calculation failed")
                return {'bootstrap_mean_auc': 0.5, 'bootstrap_std_auc': 0.0}
                
        except Exception as e:
            logger.warning(f"Bootstrap CI calculation failed: {e}")
            return {'bootstrap_mean_auc': 0.5, 'bootstrap_std_auc': 0.0}
    
    def _statistical_significance_tests(self, predictions, labels):
        """Perform statistical significance tests"""
        results = {}
        
        try:
            # Test if AUCs are significantly different from random (0.5)
            auc_scores = []
            for i in range(Config.NUM_DISEASES):
                if len(np.unique(labels[:, i])) > 1:
                    try:
                        auc = roc_auc_score(labels[:, i], predictions[:, i])
                        auc_scores.append(auc)
                    except:
                        pass
            
            if auc_scores:
                # One-sample t-test against 0.5
                t_stat, p_value = stats.ttest_1samp(auc_scores, 0.5)
                results['ttest_vs_random_t_stat'] = t_stat
                results['ttest_vs_random_p_value'] = p_value
                results['significantly_better_than_random'] = p_value < Config.STATISTICAL_SIGNIFICANCE_ALPHA
                
                logger.info(f"Statistical Tests:")
                logger.info(f"  t-test vs random (p={p_value:.4f}): {'Significant' if results['significantly_better_than_random'] else 'Not significant'}")
        
        except Exception as e:
            logger.warning(f"Statistical significance tests failed: {e}")
            results['ttest_vs_random_p_value'] = 1.0
            results['significantly_better_than_random'] = False
        
        return results
    
    def _create_visualizations(self, predictions, labels, metrics):
        """Create comprehensive visualizations"""
        try:
            self._plot_roc_curves(predictions, labels, metrics)
            self._plot_pr_curves(predictions, labels, metrics)
            self._plot_auc_distribution(metrics)
            self._plot_confusion_matrices(predictions, labels, metrics)
            self._plot_calibration_curves(predictions, labels)
        except Exception as e:
            logger.warning(f"Visualization creation failed: {e}")
    
    def _plot_roc_curves(self, predictions, labels, metrics):
        """Plot ROC curves for all diseases"""
        try:
            fig, axes = plt.subplots(4, 4, figsize=(20, 16))
            axes = axes.flatten()
            
            for i, disease in enumerate(Config.PATHOLOGY_NAMES):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                if len(np.unique(labels[:, i])) > 1:
                    fpr = metrics[f'{disease}_fpr']
                    tpr = metrics[f'{disease}_tpr']
                    auc_score = metrics[f'{disease}_auc']
                    
                    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
                    
                    # Mark optimal point
                    opt_sens = metrics[f'{disease}_optimal_sensitivity']
                    opt_spec = metrics[f'{disease}_optimal_specificity']
                    ax.plot(1 - opt_spec, opt_sens, 'ro', markersize=8, label='Optimal')
                    
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.legend(loc='lower right')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No variation\nin labels', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
                
                ax.set_title(f'{disease}', fontsize=12, fontweight='bold')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
            
            # Remove extra subplots
            for i in range(len(Config.PATHOLOGY_NAMES), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.savefig(f'{CONFIG.OUTPUT_DIR}/roc_curves_{self.architecture}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"ROC curves saved: {CONFIG.OUTPUT_DIR}/roc_curves_{self.architecture}.png")

        except Exception as e:
            logger.warning(f"ROC curve plotting failed: {e}")
    
    def _plot_pr_curves(self, predictions, labels, metrics):
        """Plot Precision-Recall curves"""
        try:
            fig, axes = plt.subplots(4, 4, figsize=(20, 16))
            axes = axes.flatten()
            
            for i, disease in enumerate(Config.PATHOLOGY_NAMES):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                if len(np.unique(labels[:, i])) > 1:
                    precision, recall, _ = precision_recall_curve(labels[:, i], predictions[:, i])
                    pr_auc = metrics[f'{disease}_pr_auc']
                    prevalence = metrics[f'{disease}_prevalence']
                    
                    ax.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.3f}')
                    ax.axhline(y=prevalence, color='k', linestyle='--', alpha=0.6, 
                              label=f'Random (Prev={prevalence:.3f})')
                    
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.legend(loc='lower left')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No variation\nin labels', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
                
                ax.set_title(f'{disease}', fontsize=12, fontweight='bold')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
            
            # Remove extra subplots
            for i in range(len(Config.PATHOLOGY_NAMES), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.savefig(f'{CONFIG.OUTPUT_DIR}/pr_curves_{self.architecture}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"PR curves saved: {CONFIG.OUTPUT_DIR}/pr_curves_{self.architecture}.png")

        except Exception as e:
            logger.warning(f"PR curve plotting failed: {e}")
    
    def _plot_auc_distribution(self, metrics):
        """Plot distribution of AUC scores"""
        try:
            auc_scores = [metrics[f'{disease}_auc'] for disease in Config.PATHOLOGY_NAMES]
            f1_scores = [metrics[f'{disease}_f1_score'] for disease in Config.PATHOLOGY_NAMES]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # AUC histogram
            ax1.hist(auc_scores, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
            ax1.axvline(metrics['mean_auc'], color='red', linestyle='--', linewidth=2, label='Mean')
            ax1.set_xlabel('AUC Score')
            ax1.set_ylabel('Number of Diseases')
            ax1.set_title('Distribution of AUC Scores')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # AUC boxplot
            ax2.boxplot(auc_scores, vert=True)
            ax2.set_ylabel('AUC Score')
            ax2.set_title('AUC Score Distribution')
            ax2.grid(True, alpha=0.3)
            
            # F1 histogram
            ax3.hist(f1_scores, bins=15, alpha=0.7, edgecolor='black', color='lightcoral')
            ax3.axvline(metrics['mean_f1'], color='red', linestyle='--', linewidth=2, label='Mean')
            ax3.set_xlabel('F1 Score')
            ax3.set_ylabel('Number of Diseases')
            ax3.set_title('Distribution of F1 Scores')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Performance summary
            ax4.bar(['AUC', 'PR-AUC', 'F1'], 
                   [metrics['mean_auc'], metrics['mean_pr_auc'], metrics['mean_f1']],
                   yerr=[metrics['std_auc'], metrics['std_pr_auc'], metrics['std_f1']],
                   capsize=5, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax4.set_ylabel('Score')
            ax4.set_title('Mean Performance Metrics')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{CONFIG.OUTPUT_DIR}/performance_distribution_{self.architecture}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Performance distribution saved: {CONFIG.OUTPUT_DIR}/performance_distribution_{self.architecture}.png")

        except Exception as e:
            logger.warning(f"Distribution plotting failed: {e}")
    
    def _plot_confusion_matrices(self, predictions, labels, metrics):
        """Plot confusion matrices for top diseases"""
        try:
            # Get top 6 diseases by AUC
            disease_aucs = [(disease, metrics[f'{disease}_auc']) 
                           for disease in Config.PATHOLOGY_NAMES]
            disease_aucs.sort(key=lambda x: x[1], reverse=True)
            top_diseases = disease_aucs[:6]
            
            if not top_diseases:
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, (disease, auc_score) in enumerate(top_diseases):
                ax = axes[idx]
                disease_idx = Config.PATHOLOGY_NAMES.index(disease)
                
                threshold = metrics[f'{disease}_optimal_threshold']
                binary_preds = (predictions[:, disease_idx] > threshold).astype(int)
                true_labels = labels[:, disease_idx].astype(int)
                
                cm = confusion_matrix(true_labels, binary_preds)
                
                # Plot confusion matrix
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black",
                               fontsize=14)
                
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                ax.set_title(f'{disease}\nAUC = {auc_score:.3f}, F1 = {metrics[f"{disease}_f1_score"]:.3f}')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Negative', 'Positive'])
                ax.set_yticklabels(['Negative', 'Positive'])
            
            plt.tight_layout()
            plt.savefig(f'{CONFIG.OUTPUT_DIR}/confusion_matrices_{self.architecture}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Confusion matrices saved: {CONFIG.OUTPUT_DIR}/confusion_matrices_{self.architecture}.png")

        except Exception as e:
            logger.warning(f"Confusion matrix plotting failed: {e}")
    
    def _plot_calibration_curves(self, predictions, labels):
        """Plot calibration curves for model predictions"""
        try:
            from sklearn.calibration import calibration_curve
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            # Select top 6 diseases by prevalence for calibration analysis
            prevalences = [(disease, np.mean(labels[:, i])) 
                          for i, disease in enumerate(Config.PATHOLOGY_NAMES)]
            prevalences.sort(key=lambda x: x[1], reverse=True)
            top_diseases = prevalences[:6]
            
            for idx, (disease, prevalence) in enumerate(top_diseases):
                ax = axes[idx]
                disease_idx = Config.PATHOLOGY_NAMES.index(disease)
                
                if len(np.unique(labels[:, disease_idx])) > 1:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        labels[:, disease_idx], predictions[:, disease_idx], n_bins=10
                    )
                    
                    ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
                           linewidth=2, label=f'{disease}')
                    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                    
                    ax.set_xlabel('Mean Predicted Probability')
                    ax.set_ylabel('Fraction of Positives')
                    ax.set_title(f'{disease} Calibration\n(Prevalence: {prevalence:.3f})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Insufficient\nvariation', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{disease} Calibration')
            
            plt.tight_layout()
            plt.savefig(f'{CONFIG.OUTPUT_DIR}/calibration_curves_{self.architecture}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Calibration curves saved: {CONFIG.OUTPUT_DIR}/calibration_curves_{self.architecture}.png")

        except Exception as e:
            logger.warning(f"Calibration curve plotting failed: {e}")
    
    def _save_predictions(self, predictions, labels, filenames, metrics):
        """Save predictions and results"""
        try:
            results_dir = f'{CONFIG.OUTPUT_DIR}'
            os.makedirs(results_dir, exist_ok=True)
            
            # Save predictions
            predictions_df = pd.DataFrame(predictions, columns=Config.PATHOLOGY_NAMES)
            predictions_df['filename'] = filenames
            predictions_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
            
            # Save labels
            labels_df = pd.DataFrame(labels, columns=Config.PATHOLOGY_NAMES)
            labels_df['filename'] = filenames
            labels_df.to_csv(os.path.join(results_dir, 'labels.csv'), index=False)
            
            # Save detailed metrics
            with open(os.path.join(results_dir, 'detailed_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Save summary report
            self._create_summary_report(metrics, results_dir)
            
            logger.info(f"Results saved to {results_dir}/")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _create_summary_report(self, metrics, results_dir):
        """Create a human-readable summary report"""
        try:
            report_path = os.path.join(results_dir, 'evaluation_summary.txt')
            
            with open(report_path, 'w') as f:
                f.write(f"CHEST X-RAY MODEL EVALUATION REPORT\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Architecture: {metrics['architecture']}\n")
                f.write(f"Evaluation Date: {metrics['evaluation_date']}\n")
                f.write(f"Total Samples: {metrics['total_samples']}\n\n")
                
                f.write(f"OVERALL PERFORMANCE\n")
                f.write(f"-" * 30 + "\n")
                f.write(f"Mean AUC: {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}\n")
                f.write(f"AUC Range: [{metrics['min_auc']:.4f}, {metrics['max_auc']:.4f}]\n")
                f.write(f"Mean PR-AUC: {metrics['mean_pr_auc']:.4f}\n")
                f.write(f"Mean F1 Score: {metrics['mean_f1']:.4f}\n")
                
                if 'auc_95_ci_lower' in metrics:
                    f.write(f"95% Confidence Interval: [{metrics['auc_95_ci_lower']:.4f}, {metrics['auc_95_ci_upper']:.4f}]\n")
                
                f.write(f"\nSTATISTICAL SIGNIFICANCE\n")
                f.write(f"-" * 30 + "\n")
                if 'significantly_better_than_random' in metrics:
                    significance = "Yes" if metrics['significantly_better_than_random'] else "No"
                    f.write(f"Significantly better than random: {significance}\n")
                    f.write(f"p-value: {metrics.get('ttest_vs_random_p_value', 'N/A'):.6f}\n")
                
                f.write(f"\nPER-DISEASE PERFORMANCE\n")
                f.write(f"-" * 30 + "\n")
                f.write(f"{'Disease':<20} {'AUC':<8} {'PR-AUC':<8} {'F1':<8} {'Prevalence':<10}\n")
                f.write(f"-" * 60 + "\n")
                
                for disease in Config.PATHOLOGY_NAMES:
                    auc = metrics[f'{disease}_auc']
                    pr_auc = metrics[f'{disease}_pr_auc']
                    f1 = metrics[f'{disease}_f1_score']
                    prev = metrics[f'{disease}_prevalence']
                    f.write(f"{disease:<20} {auc:<8.4f} {pr_auc:<8.4f} {f1:<8.4f} {prev:<10.4f}\n")
            
            logger.info(f"Summary report saved: {report_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create summary report: {e}")
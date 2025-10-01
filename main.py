import logging
import traceback
import sys
import os
from pipeline import UnifiedPipeline
from config import Config

# Configure logging with better formatting
def setup_logging():
    """Setup comprehensive logging"""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{Config.ARCHITECTURE}/training_{Config.ARCHITECTURE}.log')
        ]
    )
    
    # Set specific log levels for different components
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {Config.ARCHITECTURE}/training_{Config.ARCHITECTURE}.log")
    return logger

def print_system_info():
    """Print system information"""
    import torch
    import platform
    
    print("\n" + "="*60)
    print("CHEST X-RAY ANALYSIS SYSTEM")
    print("="*60)
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {Config.ARCHITECTURE}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Device: {Config.DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")

def validate_config():
    """Validate configuration settings"""
    issues = []
    
    # Validate architecture
    valid_architectures = list(Config.MULTIMODAL_ARCHITECTURES.keys())
    if Config.ARCHITECTURE not in valid_architectures:
        issues.append(f"Invalid architecture '{Config.ARCHITECTURE}'. Valid options: {valid_architectures}")
    
    # Validate backbone
    valid_backbones = ["resnet50", "densenet121"]
    if Config.BACKBONE not in valid_backbones:
        issues.append(f"Invalid backbone '{Config.BACKBONE}'. Valid options: {valid_backbones}")
    
    # Validate batch size
    if Config.BATCH_SIZE < 1:
        issues.append(f"Batch size must be >= 1, got {Config.BATCH_SIZE}")
    
    # Validate learning rate
    if Config.LEARNING_RATE <= 0:
        issues.append(f"Learning rate must be > 0, got {Config.LEARNING_RATE}")
    
    # Validate image size
    if Config.IMAGE_SIZE < 64:
        issues.append(f"Image size should be >= 64, got {Config.IMAGE_SIZE}")
    
    return issues

def main():
    """Main function with comprehensive error handling"""
    logger = setup_logging()
    
    try:
        print_system_info()
        
        # Validate configuration
        config_issues = validate_config()
        if config_issues:
            logger.error("Configuration validation failed:")
            for issue in config_issues:
                logger.error(f"  • {issue}")
            return False
        
        logger.info("Configuration validation passed")
        
        # Initialize and run pipeline
        pipeline = UnifiedPipeline()
        
        logger.info("Starting pipeline execution...")
        metrics, results, history = pipeline.run_pipeline()
        
        # Print success message
        print("\n" + "="*60)
        print("✅ SYSTEM COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final Mean AUC: {metrics.get('mean_auc', 0):.4f}")
        if 'auc_95_ci_lower' in metrics:
            print(f"95% Confidence Interval: [{metrics['auc_95_ci_lower']:.4f}, {metrics['auc_95_ci_upper']:.4f}]")
        
        significance = metrics.get('significantly_better_than_random', False)
        print(f"Significantly better than random: {'Yes' if significance else 'No'}")
        print("="*60 + "\n")
        
        # Print file locations
        logger.info("Generated files:")
        logger.info(f"  • Training log: training_{Config.ARCHITECTURE}.log")
        logger.info(f"  • Best model: best_model_{Config.ARCHITECTURE}.pth")
        logger.info(f"  • Results directory: results_{Config.ARCHITECTURE}/")
        logger.info(f"  • Performance plots: *_{Config.ARCHITECTURE}.png")
        logger.info(f"  • Summary CSV: performance_summary_{Config.ARCHITECTURE}.csv")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("💡 TRAINING INTERRUPTED BY USER")
        logger.info("="*60)
        logger.info("Training was stopped by user (Ctrl+C)")
        logger.info("Partial results may be available in the results directory")
        return False
        
    except Exception as e:
        logger.error("\n" + "="*60)
        logger.error("💥 CRITICAL ERROR OCCURRED")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}")
        logger.error("Full traceback:")
        traceback.print_exc()
        logger.error("="*60)
        
        # Provide helpful troubleshooting information
        logger.error("\nTroubleshooting tips:")
        logger.error("1. Check if data directory and files exist")
        logger.error("2. Verify CUDA installation if using GPU")
        logger.error("3. Ensure sufficient memory (RAM/VRAM)")
        logger.error("4. Check file permissions")
        logger.error("5. Reduce batch size if running out of memory")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
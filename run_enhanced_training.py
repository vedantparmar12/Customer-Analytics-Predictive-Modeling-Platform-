"""
Run enhanced model training to achieve 80% accuracy
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_training_enhanced import EnhancedModelTrainer
from src.logger import get_logger
import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

def main():
    try:
        logger.info("="*70)
        logger.info("ENHANCED MODEL TRAINING - TARGET 80% ACCURACY")
        logger.info("="*70)
        
        # Check if processed data exists
        if not os.path.exists("artifacts/processed_final/X_train.pkl"):
            logger.error("Processed data not found. Please run the pipeline first.")
            return
        
        # Run enhanced training
        trainer = EnhancedModelTrainer("artifacts/processed_final")
        results = trainer.run()
        
        logger.info("\nTraining completed! Check artifacts/models_final/ for enhanced models.")
        
    except Exception as e:
        logger.error(f"Error in enhanced training: {e}")

if __name__ == "__main__":
    main()
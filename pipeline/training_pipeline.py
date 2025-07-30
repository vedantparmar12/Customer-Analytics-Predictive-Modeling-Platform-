import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import CustomerDataProcessor
from src.eda import CustomerAnalyticsEDA
from src.model_training import ChurnPredictionModel
from src.recommendation_engine import RecommendationEngine
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class CustomerAnalyticsPipeline:
    def __init__(self):
        self.data_path = "data"
        self.processed_path = "artifacts/processed"
        self.models_path = "artifacts/models"
        self.eda_path = "artifacts/eda"
        
        logger.info("Customer Analytics Pipeline initialized")
    
    def run_data_processing(self):
        """Run data processing step"""
        try:
            logger.info("Starting data processing...")
            processor = CustomerDataProcessor(self.data_path, self.processed_path)
            processor.run()
            logger.info("Data processing completed")
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise CustomException("Data processing failed", e)
    
    def run_eda(self):
        """Run exploratory data analysis"""
        try:
            logger.info("Starting EDA...")
            eda = CustomerAnalyticsEDA(
                os.path.join(self.processed_path, "customer_features.csv"),
                self.eda_path
            )
            eda.run()
            logger.info("EDA completed")
        except Exception as e:
            logger.error(f"Error in EDA: {e}")
            raise CustomException("EDA failed", e)
    
    def run_model_training(self):
        """Run model training"""
        try:
            logger.info("Starting model training...")
            trainer = ChurnPredictionModel(self.processed_path, self.models_path)
            trainer.run()
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException("Model training failed", e)
    
    def run_recommendation_engine(self):
        """Run recommendation engine"""
        try:
            logger.info("Starting recommendation engine...")
            engine = RecommendationEngine(self.data_path, self.models_path)
            engine.run()
            logger.info("Recommendation engine completed")
        except Exception as e:
            logger.error(f"Error in recommendation engine: {e}")
            raise CustomException("Recommendation engine failed", e)
    
    def run_full_pipeline(self):
        """Run the complete analytics pipeline"""
        try:
            logger.info("="*50)
            logger.info("Starting Customer Analytics Pipeline")
            logger.info("="*50)
            
            # Step 1: Data Processing
            self.run_data_processing()
            
            # Step 2: EDA
            self.run_eda()
            
            # Step 3: Model Training
            self.run_model_training()
            
            # Step 4: Recommendation Engine
            self.run_recommendation_engine()
            
            logger.info("="*50)
            logger.info("Customer Analytics Pipeline Completed Successfully!")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise CustomException("Pipeline execution failed", e)

if __name__ == "__main__":
    pipeline = CustomerAnalyticsPipeline()
    pipeline.run_full_pipeline()
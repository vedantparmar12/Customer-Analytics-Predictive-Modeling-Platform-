"""
Selective Pipeline - Run specific analytics components
Allows you to choose which components to run
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from master_pipeline import MasterPipeline
from src.logger import get_logger

logger = get_logger(__name__)

class SelectivePipeline(MasterPipeline):
    """Pipeline that allows selective execution of components"""
    
    def run_selected(self, components):
        """Run only selected components"""
        
        logger.info("="*70)
        logger.info("SELECTIVE PIPELINE EXECUTION")
        logger.info("="*70)
        logger.info(f"Selected components: {', '.join(components)}")
        logger.info("="*70)
        
        results = {}
        
        # Map component names to methods
        component_map = {
            'data': self.run_data_processing,
            'eda': self.run_eda,
            'models': self.run_model_training,
            'metrics': self.run_business_metrics,
            'cohort': self.run_cohort_analysis,
            'ab': self.run_ab_testing_framework,
            'recommend': self.run_recommendation_engine,
            'nlp': self.run_nlp_analysis,
            'forecast': self.run_forecasting
        }
        
        # Run selected components
        for component in components:
            if component in component_map:
                logger.info(f"\nRunning: {component}")
                try:
                    result = component_map[component]()
                    results[component] = 'SUCCESS' if result else 'FAILED'
                except Exception as e:
                    logger.error(f"Component {component} failed: {e}")
                    results[component] = 'FAILED'
            else:
                logger.warning(f"Unknown component: {component}")
        
        # Show results
        logger.info("\n" + "="*70)
        logger.info("SELECTIVE PIPELINE RESULTS")
        logger.info("="*70)
        for component, status in results.items():
            logger.info(f"{component}: {status}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Run selective analytics pipeline')
    
    parser.add_argument(
        '--components',
        nargs='+',
        choices=['data', 'eda', 'models', 'metrics', 'cohort', 'ab', 'recommend', 'nlp', 'forecast', 'all'],
        default=['all'],
        help='Components to run'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick analysis (data + metrics + models only)'
    )
    
    args = parser.parse_args()
    
    pipeline = SelectivePipeline()
    
    if 'all' in args.components:
        # Run everything
        pipeline.run()
    elif args.quick:
        # Quick analysis
        pipeline.run_selected(['data', 'models', 'metrics'])
    else:
        # Run selected components
        pipeline.run_selected(args.components)

if __name__ == "__main__":
    main()
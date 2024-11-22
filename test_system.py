import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from config import Config
from llm_handler import LLMHandler
from pca_analyzer import PCAAnalyzer

def test_system():
    """Manual test script to verify entire system functionality"""
    
    # 1. Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting system test...")

    try:
        # 2. Initialize configuration
        config = Config()
        api_key = input("OpenAI_API_KEY").strip()
        config.set('openai_api_key', api_key)
        
        # 3. Load or create test data
        logger.info("Creating test dataset...")
        n_samples = 50
        n_features = 10
        
        # Create synthetic data
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        prompts = [f"Test prompt {i}" for i in range(n_samples)]
        
        # 4. Initialize components
        logger.info("Initializing components...")
        analyzer = PCAAnalyzer(data, prompts)
        llm_handler = LLMHandler(config)
        
        # 5. Run PCA
        logger.info("Running PCA...")
        analyzer.run_pca(n_components=3)
        
        # 6. Generate names for components
        logger.info("Generating names for components...")
        results = []
        for i in range(3):
            logger.info(f"Processing PC {i+1}...")
            pc_data = llm_handler.prepare_pc_data(analyzer, i)
            result = llm_handler.generate_name(pc_data)
            results.append(result)
            
            # Print results as they come in
            print(f"\nPC {i+1} Results:")
            print(f"Name: {result.get('name', 'Error')}")
            print(f"Explanation: {result.get('explanation', 'Error')}")
            print("-" * 50)
        
        # 7. Export results
        logger.info("Exporting results...")
        export_path = Path(f"pca_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        saved_path = llm_handler.export_results(results, export_path)
        logger.info(f"Results exported to: {saved_path}")
        
        # 8. Print summary
        print("\nTest Summary:")
        print(f"Total PCs processed: {len(results)}")
        print(f"Successful generations: {sum(1 for r in results if 'error' not in r)}")
        print(f"Failed generations: {sum(1 for r in results if 'error' in r)}")
        print(f"Results exported to: {saved_path}")
        
        return True

    except Exception as e:
        logger.error(f"An error occurred during system test: {e}")
        return False 
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

from config import Config
from llm_handler import LLMHandler

def generate_pca_names(pca_results, prompts_df, n_components):
    """Generate names for PCA components from GUI-provided PCA results
    
    Args:
        pca_results: PCA results from the GUI analysis
        prompts_df: DataFrame containing prompt information
        n_components: Number of components to name
        
    Returns:
        pandas.DataFrame: DataFrame containing generated names and explanations
    """
    try:
        # 1. Setup configuration
        config = Config()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("No API key found. Please set OPENAI_API_KEY environment variable")
        config.set('openai_api_key', api_key)

        # 2. Initialize components
        llm_handler = LLMHandler(config)

        # 3. Generate names for components
        results = []
        for i in range(n_components):
            pc_data = {
                'pc_num': i + 1,
                'top_dims': pca_results.get(f'pc_{i+1}_top_dims', ''),
                'high_prompts': pca_results.get(f'pc_{i+1}_high_prompts', ''),
                'low_prompts': pca_results.get(f'pc_{i+1}_low_prompts', '')
            }
            
            result = llm_handler.generate_name(pc_data)
            results.append(result)

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        return df_results

    except Exception as e:
        print(f"Error during PCA name generation: {str(e)}")
        return None

# Make sure the function is available for import
__all__ = ['generate_pca_names'] 
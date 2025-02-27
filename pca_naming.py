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
            print(f"\nDEBUG: PCA results received for PC{i+1}:")
            print(f"High prompts: {pca_results.get(f'pc_{i+1}_high_prompts', '')}")
            print(f"Low prompts: {pca_results.get(f'pc_{i+1}_low_prompts', '')}")
            pc_data = {
                'pc_num': i + 1,
                'high_prompts': pca_results.get(f'pc_{i+1}_high_prompts', ''),
                'low_prompts': pca_results.get(f'pc_{i+1}_low_prompts', ''),
                'additional_info': (
                    "Please identify novel and specific themes from the high and low prompts, "
                    "rather than relying on the original value dimensions. Each PC may reveal "
                    "different types of patterns, so don't feel constrained to similar themes "
                    "across components. Include both a high theme and a low theme in your analysis."
                )
            }
            
            result = llm_handler.generate_name(pc_data)
            results.append(result)

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        return df_results

    except Exception as e:
        print(f"Error during PCA name generation: {str(e)}")
        return None

def generate_ica_names(ica_results, prompts_df, n_components):
    """Generate names for ICA components from GUI-provided ICA results
    
    Args:
        ica_results: ICA results from the GUI analysis
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
            ic_data = {
                'ic_num': i + 1,
                'high_prompts': ica_results.get(f'ic_{i+1}_high_prompts', ''),
                'low_prompts': ica_results.get(f'ic_{i+1}_low_prompts', ''),
                'kurtosis': ica_results.get(f'ic_{i+1}_kurtosis', '')
            }
            
            # Add kurtosis information to the prompt
            ic_data['additional_info'] = f"This is an Independent Component with kurtosis {ic_data['kurtosis']:.3f}. " \
                                       f"High kurtosis suggests more non-Gaussian, potentially meaningful signals."
            
            result = llm_handler.generate_name(ic_data, component_type='ica')
            results.append(result)

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        return df_results

    except Exception as e:
        print(f"Error during ICA name generation: {str(e)}")
        return None

# Make sure both functions are available for import
__all__ = ['generate_pca_names', 'generate_ica_names'] 
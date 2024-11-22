import pytest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
from pathlib import Path

from config import Config
from llm_handler import LLMHandler
from pca_analyzer import PCAAnalyzer  # Assuming this is your main class

class TestPCANameGeneration:
    @pytest.fixture
    def mock_data(self):
        """Create mock PCA data"""
        # Create synthetic data
        n_samples = 100
        n_features = 10
        np.random.seed(42)
        
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        prompts = [f"Test prompt {i}" for i in range(n_samples)]
        
        return data, prompts

    @pytest.fixture
    def mock_openai_response(self):
        """Create mock OpenAI API response"""
        return Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="Name: Test Component\nExplanation: This component represents test patterns"
                    )
                )
            ]
        )

    def test_end_to_end_flow(self, mock_data, mock_openai_response, tmp_path):
        """Test the entire flow from PCA to name generation"""
        data, prompts = mock_data
        
        # Initialize components
        config = Config()
        config.set('openai_api_key', 'test_key')
        
        # Create PCA analyzer
        analyzer = PCAAnalyzer(data, prompts)
        analyzer.run_pca(n_components=3)
        
        # Initialize LLM handler with mocked API
        with patch('openai.OpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_openai_response
            
            llm_handler = LLMHandler(config)
            
            # Generate names for all components
            results = []
            for i in range(3):
                pc_data = llm_handler.prepare_pc_data(analyzer, i)
                result = llm_handler.generate_name(pc_data)
                results.append(result)
            
            # Verify results
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result['pc_num'] == i + 1
                assert 'name' in result
                assert 'explanation' in result
                assert 'error' not in result
            
            # Test export functionality
            export_path = tmp_path / 'test_results.xlsx'
            saved_path = llm_handler.export_results(results, export_path)
            
            assert Path(saved_path).exists()
            df = pd.read_excel(saved_path)
            assert len(df) == 3 
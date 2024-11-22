import pytest
from unittest.mock import Mock, patch
import time
from pathlib import Path
import pandas as pd
from openai import OpenAI, AuthenticationError

from llm_handler import LLMHandler
from config import Config

class TestLLMHandler:
    @pytest.fixture
    def config(self, tmp_path):
        """Create a test config instance"""
        config = Config()
        config.config_dir = tmp_path / '.pca_analyzer'
        config.config_file = config.config_dir / 'config.json'
        config.set('openai_api_key', 'test_key')
        return config

    @pytest.fixture
    def mock_pca_instance(self):
        """Create a mock PCA instance"""
        mock = Mock()
        mock.pca_ratings = [[0.1, 0.2], [0.3, 0.4]]
        mock.prompts = ["prompt1", "prompt2"]
        mock.pca.components_ = [[0.5, 0.6], [0.7, 0.8]]
        mock.original_dims = ["dim1", "dim2"]
        return mock

    @pytest.fixture
    def llm_handler(self, config):
        """Create a test LLM handler instance"""
        return LLMHandler(config)

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI API response"""
        return Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="Name: Test Name\nExplanation: Test explanation"
                    )
                )
            ]
        )

    def test_prepare_pc_data(self, llm_handler, mock_pca_instance):
        """Test preparation of PC data"""
        result = llm_handler.prepare_pc_data(mock_pca_instance, 0)
        
        assert result['pc_num'] == 1
        assert 'top_dims' in result
        assert 'high_prompts' in result
        assert 'low_prompts' in result

    @patch('openai.OpenAI')
    def test_generate_name_success(self, mock_openai, llm_handler, mock_openai_response):
        """Test successful name generation"""
        mock_openai.return_value.chat.completions.create.return_value = mock_openai_response
        results = []

        def callback(result):
            results.append(result)

        pc_data = {
            'pc_num': 1,
            'top_dims': 'dim1',
            'high_prompts': 'prompt1',
            'low_prompts': 'prompt2'
        }

        thread = llm_handler.generate_name(pc_data, callback=callback)
        thread.join()

        assert len(results) == 1
        assert 'name' in results[0]
        assert 'explanation' in results[0]
        assert 'error' not in results[0]

    @patch('openai.OpenAI')
    def test_generate_name_error(self, mock_openai, llm_handler):
        """Test error handling in name generation"""
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
        results = []

        def callback(result):
            results.append(result)

        pc_data = {
            'pc_num': 1,
            'top_dims': 'dim1',
            'high_prompts': 'prompt1',
            'low_prompts': 'prompt2'
        }

        thread = llm_handler.generate_name(pc_data, callback=callback)
        thread.join()

        assert len(results) == 1
        assert 'error' in results[0]
        assert 'API Error' in results[0]['error']

    def test_export_results(self, llm_handler, tmp_path):
        """Test results export functionality"""
        results = [
            {'name': 'Test1', 'explanation': 'Exp1'},
            {'name': 'Test2', 'explanation': 'Exp2'}
        ]
        export_path = tmp_path / 'test_export.xlsx'
        
        result_path = llm_handler.export_results(results, str(export_path))
        
        assert Path(result_path).exists()
        df = pd.read_excel(result_path)
        assert len(df) == 2
        assert 'name' in df.columns
        assert 'explanation' in df.columns 
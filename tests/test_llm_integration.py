import pytest
from unittest.mock import Mock, patch
import time
from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any
import asyncio
import os

from config import Config
from llm_handler import LLMHandler
from response_parser import ResponseParser

class TestLLMIntegration:
    @pytest.fixture
    def config(self, tmp_path):
        """Create a test config instance"""
        config = Config()
        config.config_dir = tmp_path / '.pca_analyzer'
        config.config_file = config.config_dir / 'config.json'
        config.set('openai_api_key', 'test_key')
        return config

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI API response"""
        return Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="Name: Test Component\nExplanation: Test explanation"
                    )
                )
            ]
        )

    @pytest.fixture
    def llm_handler(self, config):
        """Create a test LLM handler instance"""
        return LLMHandler(config)

    def test_integration_successful_response(self, llm_handler, mock_openai_response):
        """Test successful integration of LLM handler and response parser"""
        with patch('openai.OpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_openai_response
            
            pc_data = {
                'pc_num': 1,
                'top_dims': 'dim1: 0.8\ndim2: 0.6',
                'high_prompts': 'prompt1: 0.9\nprompt2: 0.8',
                'low_prompts': 'prompt3: -0.8\nprompt4: -0.7'
            }
            
            result = llm_handler.generate_name(pc_data)
            
            assert result['pc_num'] == 1
            assert result['name'] == "Test Component"
            assert result['explanation'] == "Test explanation"
            assert 'error' not in result
            assert 'timestamp' in result

    def test_integration_rate_limiting(self, llm_handler, mock_openai_response):
        """Test rate limiting during multiple requests"""
        with patch('openai.OpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_openai_response
            
            pc_data = {
                'pc_num': 1,
                'top_dims': 'dim1: 0.8',
                'high_prompts': 'prompt1: 0.9',
                'low_prompts': 'prompt2: -0.8'
            }
            
            start_time = time.time()
            results = []
            
            # Make multiple requests
            for i in range(3):
                results.append(llm_handler.generate_name(pc_data))
            
            end_time = time.time()
            
            # Verify rate limiting
            assert end_time - start_time >= 2 * llm_handler.min_request_interval
            
            # Verify all requests were successful
            assert all('name' in result for result in results)
            assert all('error' not in result for result in results)

    def test_integration_error_handling(self, llm_handler):
        """Test error handling in integration"""
        with patch('openai.OpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = Exception("API Error")
            
            pc_data = {
                'pc_num': 1,
                'top_dims': 'dim1: 0.8',
                'high_prompts': 'prompt1: 0.9',
                'low_prompts': 'prompt2: -0.8'
            }
            
            result = llm_handler.generate_name(pc_data)
            
            assert 'error' in result
            assert 'API Error' in result['error']
            assert result['pc_num'] == 1

    def test_integration_malformed_response(self, llm_handler):
        """Test handling of malformed API response"""
        malformed_response = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="Invalid format response"
                    )
                )
            ]
        )
        
        with patch('openai.OpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = malformed_response
            
            pc_data = {
                'pc_num': 1,
                'top_dims': 'dim1: 0.8',
                'high_prompts': 'prompt1: 0.9',
                'low_prompts': 'prompt2: -0.8'
            }
            
            result = llm_handler.generate_name(pc_data)
            
            assert 'error' in result
            assert "missing 'Name:' section" in result['error'] 

class TestLLMAsyncIntegration:
    @pytest.fixture
    def mock_async_openai_response(self):
        """Create a mock async OpenAI API response"""
        return [Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=f"Name: Test Component {i}\nExplanation: Test explanation {i}"
                    )
                )
            ]
        ) for i in range(3)]

    @pytest.mark.asyncio
    async def test_async_batch_processing(self, llm_handler, mock_async_openai_response):
        """Test concurrent processing of multiple PCs"""
        with patch('openai.AsyncOpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = mock_async_openai_response
            
            pc_data_list = [
                {
                    'pc_num': i,
                    'top_dims': f'dim{i}: 0.8',
                    'high_prompts': f'prompt{i}: 0.9',
                    'low_prompts': f'prompt{i}: -0.8'
                }
                for i in range(3)
            ]
            
            results = await llm_handler.generate_names_concurrent(pc_data_list)
            
            assert len(results) == 3
            assert all('name' in result for result in results)
            assert all('error' not in result for result in results)
            assert [result['pc_num'] for result in results] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_async_rate_limiting(self, llm_handler, mock_async_openai_response):
        """Test rate limiting in async batch processing"""
        with patch('openai.AsyncOpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = mock_async_openai_response
            
            pc_data_list = [{'pc_num': i} for i in range(5)]
            
            start_time = time.time()
            results = await llm_handler.generate_names_concurrent(pc_data_list)
            end_time = time.time()
            
            # Verify rate limiting across batches
            batch_time = end_time - start_time
            expected_min_time = (len(pc_data_list) // 3) * llm_handler.min_request_interval
            assert batch_time >= expected_min_time

    @pytest.mark.asyncio
    async def test_async_error_handling(self, llm_handler):
        """Test error handling in async batch processing"""
        with patch('openai.AsyncOpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = [
                Mock(choices=[Mock(message=Mock(content="Name: Test1\nExplanation: Test1"))]),
                Exception("API Error"),
                Mock(choices=[Mock(message=Mock(content="Name: Test3\nExplanation: Test3"))])
            ]
            
            pc_data_list = [{'pc_num': i} for i in range(3)]
            results = await llm_handler.generate_names_concurrent(pc_data_list)
            
            assert len(results) == 3
            assert 'error' not in results[0]
            assert 'error' in results[1]
            assert 'error' not in results[2]

    @pytest.mark.asyncio
    async def test_async_batch_order_preservation(self, llm_handler, mock_async_openai_response):
        """Test that batch results maintain original order"""
        with patch('openai.AsyncOpenAI') as mock_client:
            # Simulate varying response times
            async def delayed_response(i):
                await asyncio.sleep(0.1 * (3 - i))  # Reverse delay
                return mock_async_openai_response[i]
            
            mock_client.return_value.chat.completions.create.side_effect = \
                lambda **kwargs: delayed_response(kwargs['messages'][1]['content'].split()[-1])
            
            pc_data_list = [
                {'pc_num': i, 'content': f'test {i}'} 
                for i in range(3)
            ]
            
            results = await llm_handler.generate_names_concurrent(pc_data_list)
            
            # Verify order preservation
            assert [result['pc_num'] for result in results] == [0, 1, 2] 

def test_integration():
    """Test full integration of all components"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No API key found. Please set OPENAI_API_KEY environment variable")
        return False
        
    try:
        # 1. Setup configuration
        print("1. Setting up configuration...")
        config = Config()
        config.set('openai_api_key', api_key)  # Use the actual API key from environment
        config.set('model', 'gpt-4')
        print("   ✓ Configuration initialized")
    except Exception as e:
        print(f"Failed to initialize configuration: {e}")
        return False 
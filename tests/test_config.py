import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch
from openai import OpenAI, AuthenticationError

from config import Config, APIKeyError

class TestConfig:
    @pytest.fixture
    def config(self, tmp_path):
        """Create a test config instance with temporary directory"""
        config = Config()
        config.config_dir = tmp_path / '.pca_analyzer'
        config.config_file = config.config_dir / 'config.json'
        return config

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client"""
        with patch('openai.OpenAI') as mock_client:
            # Mock successful API validation
            instance = mock_client.return_value
            instance.models.list.return_value = ['gpt-4', 'gpt-3.5-turbo']
            yield mock_client

    def test_validate_api_key_empty(self, config):
        """Test validation of empty API key"""
        is_valid, error = config.validate_api_key()
        assert not is_valid
        assert error == "API key is not set"

    def test_validate_api_key_invalid(self, config, mock_openai_client):
        """Test validation of invalid API key"""
        # Setup mock to raise AuthenticationError
        mock_openai_client.return_value.models.list.side_effect = AuthenticationError("Invalid API key")
        
        config.set('openai_api_key', 'invalid_key')
        is_valid, error = config.validate_api_key()
        assert not is_valid
        assert error == "Invalid API key"

    def test_validate_api_key_valid(self, config, mock_openai_client):
        """Test validation of valid API key"""
        config.set('openai_api_key', 'valid_key')
        is_valid, error = config.validate_api_key()
        assert is_valid
        assert error is None

    def test_set_model_params_valid(self, config):
        """Test setting valid model parameters"""
        params = {
            'temperature': 0.5,
            'max_tokens': 100,
            'model': 'gpt-4'
        }
        success, error = config.set_model_params(params)
        assert success
        assert error is None
        
        # Verify saved values
        assert config.get('temperature') == 0.5
        assert config.get('max_tokens') == 100
        assert config.get('model') == 'gpt-4'

    def test_set_model_params_invalid_temperature(self, config):
        """Test setting invalid temperature"""
        params = {'temperature': 3.0}  # Invalid: > 2.0
        success, error = config.set_model_params(params)
        assert not success
        assert "Temperature must be between 0 and 2" in error

    def test_set_model_params_invalid_tokens(self, config):
        """Test setting invalid max_tokens"""
        params = {'max_tokens': 5000}  # Invalid: > 4096
        success, error = config.set_model_params(params)
        assert not success
        assert "max_tokens must be between 1 and 4096" in error

    def test_set_model_params_invalid_model(self, config):
        """Test setting invalid model"""
        params = {'model': 'invalid-model'}
        success, error = config.set_model_params(params)
        assert not success
        assert "Model must be one of" in error

    def test_get_model_params_default(self, config):
        """Test getting default model parameters"""
        params = config.get_model_params()
        assert params['temperature'] == 0.7
        assert params['max_tokens'] == 200
        assert params['model'] == 'gpt-4'

    def test_validate_model_params(self, config):
        """Test validation of current model parameters"""
        # First set some valid parameters
        config.set_model_params({
            'temperature': 0.5,
            'max_tokens': 100,
            'model': 'gpt-4'
        })
        
        # Validate them
        is_valid, error = config.validate_model_params()
        assert is_valid
        assert error is None 
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI, AuthenticationError

class ConfigError(Exception):
    """Base class for configuration errors"""
    pass

class APIKeyError(ConfigError):
    """Raised when there are issues with the API key"""
    pass

DEFAULT_PROMPT_TEMPLATE = """You are a data scientist specializing in interpreting Principal Component Analysis (PCA) results. 
Your task is to generate an intuitive name and explanation for a principal component based on its characteristics.

Principal Component #{pc_num}
Top Contributing Dimensions:
{top_dims}

High-Loading Prompts:
{high_prompts}

Low-Loading Prompts:
{low_prompts}

Please analyze these patterns and provide:
1. A concise, descriptive name that captures the essence of this component
2. A brief explanation of why this name fits the pattern

Format your response exactly as:
Name: [your suggested name]
Explanation: [your explanation]"""

class Config:
    def __init__(self):
        self.config_dir = Path.home() / '.pca_analyzer'
        self.config_file = self.config_dir / 'config.json'
        self.api_key_file = Path('.env')
        self.settings = {
            'openai_api_key': None,
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 200,
            'default_prompt_template': DEFAULT_PROMPT_TEMPLATE
        }
        self.load_config()
        self.load_api_key()

    def load_config(self):
        """Load configuration from file or create default"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.settings = {**self.settings, **json.load(f)}
            else:
                self.settings = self.settings
                self.save_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            self.settings = self.settings

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key: str, default=None):
        """Get a configuration value"""
        return self.settings.get(key, default)

    def set(self, key: str, value):
        """Set a configuration value"""
        self.settings[key] = value

    def validate_api_key(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that the API key is set and working.
        Returns: (is_valid: bool, error_message: Optional[str])
        """
        api_key = self.get('openai_api_key')
        
        if not api_key or not api_key.strip():
            return False, "API key is not set"
            
        try:
            # Test the API key with a minimal API call
            client = OpenAI(api_key=api_key)
            client.models.list()
            return True, None
        except AuthenticationError:
            return False, "Invalid API key"
        except Exception as e:
            return False, f"Error validating API key: {str(e)}"

    def set_api_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """
        Set and validate the API key
        Returns: (success: bool, error_message: Optional[str])
        """
        if not api_key or not api_key.strip():
            return False, "API key cannot be empty"
            
        try:
            # Validate the new key before saving
            client = OpenAI(api_key=api_key.strip())
            client.models.list()
            
            # If validation succeeds, save the key
            self.set('openai_api_key', api_key.strip())
            return True, None
        except AuthenticationError:
            return False, "Invalid API key"
        except Exception as e:
            return False, f"Error setting API key: {str(e)}"

    def get_openai_client(self) -> Tuple[Optional[OpenAI], Optional[str]]:
        """
        Get an initialized OpenAI client
        Returns: (client: Optional[OpenAI], error_message: Optional[str])
        """
        api_key = self.get('openai_api_key')
        if not api_key or not api_key.strip():
            return None, "API key is not set"
            
        try:
            client = OpenAI(api_key=api_key)
            return client, None
        except Exception as e:
            return None, f"Error initializing OpenAI client: {str(e)}"

    def set_model_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Set model parameters (temperature, max_tokens, model)
        Returns: (success: bool, error_message: Optional[str])
        """
        try:
            # Validate temperature
            if 'temperature' in params:
                temp = float(params['temperature'])
                if not 0 <= temp <= 2:
                    return False, "Temperature must be between 0 and 2"
                self.settings['temperature'] = temp
            
            # Validate max_tokens
            if 'max_tokens' in params:
                tokens = int(params['max_tokens'])
                if not 0 < tokens <= 4096:
                    return False, "max_tokens must be between 1 and 4096"
                self.settings['max_tokens'] = tokens
            
            # Validate model
            if 'model' in params:
                valid_models = ['gpt-4', 'gpt-4-turbo-preview', 'gpt-3.5-turbo']
                if params['model'] not in valid_models:
                    return False, f"Model must be one of: {', '.join(valid_models)}"
                self.settings['model'] = params['model']
            
            self.save_config()
            return True, None
        
        except ValueError as e:
            return False, f"Invalid parameter value: {str(e)}"
        except Exception as e:
            return False, f"Error setting model parameters: {str(e)}"

    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return {
            'temperature': self.settings.get('temperature', 0.7),
            'max_tokens': self.settings.get('max_tokens', 200),
            'model': self.settings.get('model', 'gpt-4')
        }

    def validate_model_params(self) -> Tuple[bool, Optional[str]]:
        """
        Validate current model parameters
        Returns: (is_valid: bool, error_message: Optional[str])
        """
        params = self.get_model_params()
        return self.set_model_params(params)

    def load_api_key(self):
        """Load API key from file"""
        try:
            if self.api_key_file.exists():
                with open(self.api_key_file, 'r') as f:
                    api_key = f.read().strip()
                    if api_key:
                        self.settings['openai_api_key'] = api_key
        except Exception as e:
            print(f"Error loading API key: {e}")
  
from openai import OpenAI, RateLimitError, APIError, AsyncOpenAI
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from pathlib import Path
import logging
from response_parser import ResponseParser
import tiktoken
from dataclasses import dataclass
from datetime import datetime
from error_handler import with_error_handling
from logger import PCALogger
import json

@dataclass
class APIUsage:
    """Track API usage and costs"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    timestamp: datetime

class LLMHandlerError(Exception):
    """Base exception class for LLM handler errors"""
    pass

class RateLimitExceeded(LLMHandlerError):
    """Raised when API rate limit is exceeded"""
    pass

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class LLMHandler:
    # Cost per 1k tokens (as of latest OpenAI pricing)
    COST_PER_1K_TOKENS = {
        'gpt-4': {'prompt': 0.03, 'completion': 0.06},
        'gpt-4-turbo-preview': {'prompt': 0.01, 'completion': 0.03},
        'gpt-3.5-turbo': {'prompt': 0.0005, 'completion': 0.0015}
    }

    def __init__(self, config):
        """Initialize both sync and async clients"""
        self.config = config
        self.api_key = config.get('openai_api_key')
        self.sync_client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds
        self.max_retries = 3
        self.retry_delay = 2.0  # Base delay for exponential backoff
        self.usage_log = []
        
        # Setup logging
        self.logger = PCALogger()
        self.parser = ResponseParser()

        # Initialize tokenizer
        self.model = config.get('model', 'gpt-4')
        self.encoding = tiktoken.encoding_for_model(self.model)

        self.batch_size = 3  # Maximum concurrent requests

        self.json_encoder = DateTimeEncoder()

    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _handle_api_error(self, e: Exception, attempt: int) -> float:
        """Handle API errors and determine retry delay"""
        if isinstance(e, RateLimitError):
            delay = self.retry_delay * (2 ** attempt)
            self.logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds")
            return delay
        elif isinstance(e, APIError):
            if e.status_code in [500, 502, 503, 504]:
                delay = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"API error {e.status_code}. Retrying in {delay} seconds")
                return delay
        raise LLMHandlerError(f"API error: {str(e)}")

    def count_tokens(self, messages: list) -> int:
        """Count tokens in messages before API call"""
        try:
            num_tokens = 0
            for message in messages:
                # Count message content tokens
                num_tokens += len(self.encoding.encode(message['content']))
                # Add overhead for message format
                num_tokens += 4  # Format overhead per message
            num_tokens += 2  # Conversation format overhead
            return num_tokens
        except Exception as e:
            self.logger.warning(f"Error counting tokens: {str(e)}")
            return 0

    def calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate estimated cost based on token usage"""
        try:
            cost_rates = self.COST_PER_1K_TOKENS.get(self.model, 
                        self.COST_PER_1K_TOKENS['gpt-4'])
            
            prompt_cost = (usage['prompt_tokens'] / 1000) * cost_rates['prompt']
            completion_cost = (usage['completion_tokens'] / 1000) * cost_rates['completion']
            
            return prompt_cost + completion_cost
        except Exception as e:
            self.logger.warning(f"Error calculating cost: {str(e)}")
            return 0.0

    def log_usage(self, usage: Dict[str, int]) -> None:
        """Log API usage and costs"""
        try:
            cost = self.calculate_cost(usage)
            api_usage = APIUsage(
                prompt_tokens=usage['prompt_tokens'],
                completion_tokens=usage['completion_tokens'],
                total_tokens=usage['total_tokens'],
                estimated_cost=cost,
                timestamp=datetime.now()
            )
            self.usage_log.append(api_usage)
            
            self.logger.info(
                f"API Call: {usage['total_tokens']} tokens "
                f"(${cost:.4f}) - Model: {self.model}"
            )
        except Exception as e:
            self.logger.error(f"Error logging usage: {str(e)}")

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of API usage and costs"""
        try:
            total_tokens = sum(usage.total_tokens for usage in self.usage_log)
            total_cost = sum(usage.estimated_cost for usage in self.usage_log)
            total_calls = len(self.usage_log)
            
            return {
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'total_calls': total_calls,
                'average_tokens_per_call': total_tokens / total_calls if total_calls else 0,
                'average_cost_per_call': total_cost / total_calls if total_calls else 0
            }
        except Exception as e:
            self.logger.error(f"Error generating usage summary: {str(e)}")
            return {}

    @with_error_handling
    def generate_name(self, component_data, component_type='pca'):
        """Generate a name for a component using OpenAI API
        
        Args:
            component_data: Dictionary containing component information
            component_type: String indicating the type of component ('pca' or 'ica')
        """
        try:
            if component_type == 'pca':
                prompt = self._create_pca_prompt(component_data)
            else:  # ica
                prompt = self._create_ica_prompt(component_data)

            response = self.sync_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that names statistical components."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            # Parse the response using parse_name_response instead of parse_response
            name, explanation, error = self.parser.parse_name_response(response.choices[0].message.content)
            
            if error:
                print(f"Error parsing response: {error}")
                return None
            
            # Format the result
            result = {
                'component_number': (
                    component_data.get('pc_num') if component_type == 'pca' 
                    else component_data.get('ic_num')
                ),
                'component_name': name,
                'description': explanation
            }
            
            return result

        except Exception as e:
            print(f"Error generating name: {str(e)}")
            return None

    def _create_pca_prompt(self, pc_data):
        """Create prompt for PCA component naming"""
        return f"""Please analyze this Principal Component and provide a concise name and explanation.

Component number: {pc_data['pc_num']}
Top contributing dimensions: {pc_data['top_dims']}
High-loading prompts: {pc_data['high_prompts']}
Low-loading prompts: {pc_data['low_prompts']}

Please provide:
1. A short, descriptive name (2-6 words)
2. A brief explanation of what this component might represent

Format your response as:
Name: [your suggested name]
Explanation: [your explanation]"""

    def _create_ica_prompt(self, ic_data):
        """Create prompt for ICA component naming"""
        return f"""Please analyze this Independent Component and provide a concise name and explanation.

Component number: {ic_data['ic_num']}
Top contributing dimensions: {ic_data['top_dims']}
High-loading prompts: {ic_data['high_prompts']}
Low-loading prompts: {ic_data['low_prompts']}
{ic_data.get('additional_info', '')}

Please provide:
1. A short, descriptive name (2-6 words) that captures the independent signal this component might represent
2. A brief explanation of what this independent signal might represent, considering that ICA finds statistically independent sources

Format your response as:
Name: [your suggested name]
Explanation: [your explanation]"""

    def prepare_pc_data(self, pca_instance, pc_index: int) -> Dict[str, Any]:
        """Prepare data about a PC for the LLM prompt"""
        n_examples = self.config.get('n_examples', 5)
        n_top_dims = self.config.get('n_top_dims', 5)

        # Get PC scores
        pc_scores = pca_instance.pca_ratings[:, pc_index]
        
        # Get top and bottom examples
        top_indices = pc_scores.argsort()[-n_examples:][::-1]
        bottom_indices = pc_scores.argsort()[:n_examples]
        
        # Get prompts and scores
        high_prompts = [
            f"{pca_instance.prompts[idx]}: {pc_scores[idx]:.3f}"
            for idx in top_indices
        ]
        low_prompts = [
            f"{pca_instance.prompts[idx]}: {pc_scores[idx]:.3f}"
            for idx in bottom_indices
        ]
        
        # Get top contributing dimensions
        loadings = pca_instance.pca.components_[pc_index]
        top_dim_indices = abs(loadings).argsort()[-n_top_dims:][::-1]
        top_dims = [
            f"{pca_instance.original_dims[idx]}: {loadings[idx]:.3f}"
            for idx in top_dim_indices
        ]
        
        return {
            'pc_num': pc_index + 1,
            'top_dims': '\n'.join(top_dims),
            'high_prompts': '\n'.join(high_prompts),
            'low_prompts': '\n'.join(low_prompts)
        }

    async def _generate_single(self, pc_data: Dict[str, Any], custom_prompt: str = None) -> Dict[str, Any]:
        """Generate a single name asynchronously"""
        try:
            prompt_template = custom_prompt or self.config.get('default_prompt_template')
            prompt = prompt_template.format(**pc_data)
            
            response = await self.async_client.chat.completions.create(
                model=self.config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant specializing in analyzing and naming statistical patterns."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 200)
            )
            
            content = response.choices[0].message.content
            name, explanation, error = self.parser.parse_name_response(content)
            
            if error:
                return {
                    'pc_num': pc_data['pc_num'],
                    'error': error
                }
                
            return {
                'pc_num': pc_data['pc_num'],
                'name': name,
                'explanation': explanation,
                'raw_response': content,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'pc_num': pc_data['pc_num'],
                'error': f'Error generating name: {str(e)}'
            }

    async def generate_names_concurrent(self, pc_data_list: List[Dict[str, Any]], 
                                     custom_prompt: str = None) -> List[Dict[str, Any]]:
        """Generate names for multiple PCs concurrently with batching"""
        results = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(pc_data_list), self.batch_size):
            batch = pc_data_list[i:i + self.batch_size]
            batch_tasks = [
                self._generate_single(pc_data, custom_prompt) 
                for pc_data in batch
            ]
            
            # Run batch concurrently
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Add delay between batches if not the last batch
            if i + self.batch_size < len(pc_data_list):
                await asyncio.sleep(1.0)  # Rate limiting delay
        
        # Sort results by PC number to maintain order
        return sorted(results, key=lambda x: x.get('pc_num', 0))

    def generate_names_batch(self, pc_data_list: List[Dict[str, Any]], 
                           custom_prompt: str = None) -> List[Dict[str, Any]]:
        """Synchronous wrapper for concurrent name generation"""
        return asyncio.run(self.generate_names_concurrent(pc_data_list, custom_prompt))

    def export_results(self, results: list, export_path: str = None) -> str:
        """Export generated names and explanations"""
        if export_path is None:
            export_path = Path.home() / 'Downloads' / 'pc_names.xlsx'
        
        df = pd.DataFrame(results)
        df.to_excel(export_path, index=False)
        return str(export_path)

    def generate_value_dimension_definitions(self, dimensions: List[str]) -> List[Dict[str, str]]:
        """Generate definitions for value dimensions using the LLM.
        
        Args:
            dimensions: List of value dimension names needing definitions
            
        Returns:
            List of dictionaries containing the dimension name and generated definition
        """
        try:
            results = []
            
            for dimension in dimensions:
                prompt = self._create_dimension_definition_prompt(dimension)
                
                response = self.sync_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specializing in defining value dimensions and design principles."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                definition = response.choices[0].message.content.strip()
                
                results.append({
                    'dimension': dimension,
                    'definition': definition
                })
                
                # Enforce rate limiting between requests
                self._enforce_rate_limit()
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating definitions: {str(e)}")
            raise

    def _create_dimension_definition_prompt(self, dimension: str) -> str:
        """Create prompt for generating a value dimension definition"""
        return f"""Please provide a clear, concise definition for the following value dimension or design principle:

Value Dimension: {dimension}

Requirements:
1. Definition should be 1-2 sentences
2. Focus on how this dimension relates to design or user experience
3. Use clear, professional language
4. Avoid jargon unless necessary

Please provide only the definition without any additional formatting or labels."""
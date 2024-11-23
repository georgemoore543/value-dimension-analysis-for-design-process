import pytest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
from pathlib import Path
import os

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

    def test_integration(self):
        """Test full integration of all components"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  No API key found. Please set OPENAI_API_KEY environment variable")
            return False
        
        try:
            # 1. Setup configuration
            print("1. Setting up configuration...")
            config = Config()
            config.set('openai_api_key', api_key)  # Keep existing API config
            config.set('model', 'gpt-4')
            print("   ✓ Configuration initialized")

            # 2. Initialize components with detailed error checking
            print("\n2. Initializing components...")
            try:
                llm_handler = LLMHandler(config)
                parser = ResponseParser()
                print("   ✓ Components initialized")
            except Exception as e:
                print(f"   ❌ Component initialization failed: {str(e)}")
                raise

            # 3. Prepare test data with validation
            print("\n3. Preparing test data...")
            pc_data = {
                'pc_num': 1,
                'top_dims': 'dimension1: 0.8\ndimension2: 0.6',
                'high_prompts': 'prompt1: 0.9\nprompt2: 0.8',
                'low_prompts': 'prompt3: -0.8\nprompt4: -0.7'
            }
            assert all(k in pc_data for k in ['pc_num', 'top_dims', 'high_prompts', 'low_prompts']), "Missing required fields in test data"
            print("   ✓ Test data prepared")

            # 4. Test full workflow with detailed result validation
            print("\n4. Testing full workflow...")
            result = llm_handler.generate_name(pc_data)
            
            # Add detailed result validation
            print("\n5. Validating results...")
            if not result:
                raise ValueError("No result returned from generate_name")
            
            required_fields = ['pc_num', 'name', 'explanation', 'timestamp']
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                raise ValueError(f"Missing required fields in result: {missing_fields}")
            
            print("   ✓ Result validation complete")
            print(f"   Result structure: {list(result.keys())}")
            print(f"   PC Number: {result['pc_num']}")
            print(f"   Generated Name: {result['name']}")

            print("\n✓ All integration tests passed!")
            return True

        except Exception as e:
            print(f"\n❌ Error during integration test: {str(e)}")
            import traceback
            print("\nFull error traceback:")
            print(traceback.format_exc())
            return False

if __name__ == "__main__":
    success = test_integration()
    print(f"\nIntegration test {'succeeded' if success else 'failed'}") 
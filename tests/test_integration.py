import pytest
import sys
import os
from unittest.mock import Mock

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from llm_handler import LLMHandler
from response_parser import ResponseParser

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

if __name__ == "__main__":
    success = test_integration()
    print(f"\nIntegration test {'succeeded' if success else 'failed'}") 
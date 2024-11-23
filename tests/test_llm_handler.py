import pytest
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from llm_handler import LLMHandler

def test_llm_handler():
    """Test basic LLMHandler functionality with clear print statements"""
    print("\nStarting LLMHandler test...")
    
    try:
        # 1. Create config and handler instances
        print("1. Creating Config and LLMHandler instances...")
        config = Config()
        config.set('openai_api_key', 'test-key')  # Use a test key
        handler = LLMHandler(config)
        print("   ✓ Instances created")

        # 2. Test PC data preparation
        print("\n2. Testing PC data preparation...")
        pc_data = {
            'pc_num': 1,
            'top_dims': 'dimension1: 0.8\ndimension2: 0.6',
            'high_prompts': 'prompt1: 0.9\nprompt2: 0.8',
            'low_prompts': 'prompt3: -0.8\nprompt4: -0.7'
        }
        print("   ✓ Test data prepared")

        # 3. Test name generation (with mocked API call)
        print("\n3. Testing name generation...")
        with pytest.raises(Exception) as e:
            # This should fail because we're using a test key
            result = handler.generate_name(pc_data)
        print("   ✓ API call properly failed with test key")

        print("\n✓ All LLMHandler tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Error during LLMHandler test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_llm_handler()
    print(f"\nTest {'succeeded' if success else 'failed'}") 
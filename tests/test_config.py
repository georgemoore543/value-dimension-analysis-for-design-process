import pytest
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def test_config():
    """Test basic Config functionality with clear print statements"""
    print("\nStarting Config test...")
    
    try:
        # 1. Create config instance
        print("1. Creating Config instance...")
        config = Config()
        print("   ✓ Config instance created")

        # 2. Test default values
        print("\n2. Testing default values...")
        print(f"   Model: {config.get('model')}")
        print(f"   Temperature: {config.get('temperature')}")
        print(f"   Max tokens: {config.get('max_tokens')}")
        
        # 3. Test setting a value
        print("\n3. Testing value setting...")
        config.set('test_key', 'test_value')
        value = config.get('test_key')
        print(f"   Set and retrieved value: {value}")
        
        # 4. Test getting template
        print("\n4. Testing prompt template...")
        template = config.get('default_prompt_template')
        print(f"   Template exists: {template is not None}")
        print(f"   Template length: {len(template) if template else 0} characters")

        print("\n✓ All config tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Error during config test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_config()
    print(f"\nTest {'succeeded' if success else 'failed'}")
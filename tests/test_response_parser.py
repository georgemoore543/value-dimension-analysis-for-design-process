import pytest
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from response_parser import ResponseParser

def test_response_parser():
    """Test basic ResponseParser functionality with clear print statements"""
    print("\nStarting ResponseParser test...")
    
    try:
        # 1. Create parser instance
        print("1. Creating ResponseParser instance...")
        parser = ResponseParser()
        print("   ✓ Parser instance created")

        # 2. Test valid response parsing
        print("\n2. Testing valid response parsing...")
        valid_response = """Name: Test Component
Explanation: This is a test explanation"""
        name, explanation, error = parser.parse_name_response(valid_response)
        assert name == "Test Component"
        assert explanation == "This is a test explanation"
        assert error is None
        print("   ✓ Valid response parsed correctly")

        # 3. Test invalid response parsing
        print("\n3. Testing invalid response parsing...")
        invalid_responses = [
            "",  # Empty
            "Invalid format",  # Wrong format
            "Name:",  # Missing explanation
            "Explanation: test"  # Missing name
        ]
        
        for resp in invalid_responses:
            name, explanation, error = parser.parse_name_response(resp)
            assert error is not None
            assert name is None
            assert explanation is None
        print("   ✓ Invalid responses handled correctly")

        # 4. Test result formatting
        print("\n4. Testing result formatting...")
        result = parser.format_result(
            pc_num=1,
            name="Test Component",
            explanation="Test explanation"
        )
        assert result['pc_num'] == 1
        assert result['name'] == "Test Component"
        assert result['explanation'] == "Test explanation"
        assert 'timestamp' in result
        print("   ✓ Result formatted correctly")

        print("\n✓ All ResponseParser tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Error during ResponseParser test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_response_parser()
    print(f"\nTest {'succeeded' if success else 'failed'}") 
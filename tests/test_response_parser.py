import pytest
from ..response_parser import ResponseParser
import time

class TestResponseParser:
    @pytest.fixture
    def parser(self):
        """Create a test response parser instance"""
        return ResponseParser()

    def test_parse_valid_response(self, parser):
        """Test parsing a valid response"""
        content = """Name: Test Component
Explanation: This is a test explanation"""
        
        name, explanation, error = parser.parse_name_response(content)
        
        assert name == "Test Component"
        assert explanation == "This is a test explanation"
        assert error is None

    def test_parse_invalid_response_no_name(self, parser):
        """Test parsing response without Name section"""
        content = "Explanation: This is a test explanation"
        
        name, explanation, error = parser.parse_name_response(content)
        
        assert name is None
        assert explanation is None
        assert "missing 'Name:' section" in error

    def test_parse_invalid_response_no_explanation(self, parser):
        """Test parsing response without Explanation section"""
        content = "Name: Test Component"
        
        name, explanation, error = parser.parse_name_response(content)
        
        assert name is None
        assert explanation is None
        assert "missing 'Explanation:' section" in error

    def test_parse_empty_response(self, parser):
        """Test parsing empty response"""
        content = ""
        
        name, explanation, error = parser.parse_name_response(content)
        
        assert name is None
        assert explanation is None
        assert "Empty or invalid response content" in error

    def test_parse_none_response(self, parser):
        """Test parsing None response"""
        content = None
        
        name, explanation, error = parser.parse_name_response(content)
        
        assert name is None
        assert explanation is None
        assert "Empty or invalid response content" in error

    def test_parse_malformed_response(self, parser):
        """Test parsing malformed response"""
        content = "Name: \nExplanation: "
        
        name, explanation, error = parser.parse_name_response(content)
        
        assert name is None
        assert explanation is None
        assert "Empty name in response" in error

    def test_format_result_success(self, parser):
        """Test formatting successful result"""
        result = parser.format_result(
            pc_num=1,
            name="Test Component",
            explanation="Test explanation"
        )
        
        assert result['pc_num'] == 1
        assert result['name'] == "Test Component"
        assert result['explanation'] == "Test explanation"
        assert 'error' not in result
        assert 'timestamp' in result

    def test_format_result_error(self, parser):
        """Test formatting error result"""
        result = parser.format_result(
            pc_num=1,
            name=None,
            explanation=None,
            error="Test error"
        )
        
        assert result['pc_num'] == 1
        assert 'name' not in result
        assert 'explanation' not in result
        assert result['error'] == "Test error"
        assert 'timestamp' in result 
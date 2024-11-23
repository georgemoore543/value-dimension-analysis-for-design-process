import time
from datetime import datetime
from typing import Optional

class ResponseParser:
    def parse_name_response(self, content):
        if not content:
            return None, None, "Empty or invalid response content"
            
        if "Name:" not in content:
            return None, None, "Response missing 'Name:' section"
            
        if "Explanation:" not in content:
            return None, None, "Response missing 'Explanation:' section"
            
        try:
            name = content.split("Name:")[1].split("Explanation:")[0].strip()
            explanation = content.split("Explanation:")[1].strip()
            
            if not name:
                return None, None, "Empty name in response"
                
            return name, explanation, None
            
        except Exception:
            return None, None, "Failed to parse response"

    def format_result(self, pc_num: int, name: Optional[str] = None, 
                     explanation: Optional[str] = None, error: Optional[str] = None) -> dict:
        """Format the parsed response into a standardized result dictionary"""
        result = {
            'pc_num': pc_num,
            'timestamp': datetime.now().isoformat()  # Store as string instead of datetime object
        }
        
        if error:
            result['error'] = error
            return result
        
        if name and explanation:
            result.update({
                'name': name,
                'explanation': explanation
            })
            
        return result 
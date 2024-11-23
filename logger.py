import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class PCALogger:
    """Logger for PCA name generation process"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.log_dir = log_dir or Path.home() / '.pca_analyzer' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file handler for detailed logging
        self.file_handler = logging.FileHandler(
            self.log_dir / f'pca_analyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Set up console handler for important messages
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        
        # Configure logger
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    def log_api_call(self, pc_data: Dict[str, Any], response: Dict[str, Any]):
        """Log API call details with datetime handling"""
        try:
            self.logger.info(f"Generated name for PC {pc_data['pc_num']}")
            self.logger.debug(
                f"API Response: {json.dumps(response, indent=2, cls=DateTimeEncoder)}"
            )
        except Exception as e:
            self.logger.error(f"Error logging API call: {str(e)}")

    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error details"""
        self.logger.error(
            f"Error: {str(error)}\nContext: {json.dumps(context, indent=2)}",
            exc_info=True
        )

    def log_performance(self, start_time: float, end_time: float, pc_count: int):
        """Log performance metrics"""
        duration = end_time - start_time
        avg_time = duration / pc_count if pc_count > 0 else 0
        
        self.logger.info(
            f"Performance: {pc_count} PCs processed in {duration:.2f}s "
            f"(avg: {avg_time:.2f}s per PC)"
        ) 
"""Configuration management for NoteViz."""
import os
from pathlib import Path
from typing import Optional

class Config:
    """Global configuration for NoteViz."""
    
    def __init__(self):
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.data_dir: Path = Path(os.getenv("NOTEVIZ_DATA_DIR", "data"))
        self.cache_dir: Path = Path(os.getenv("NOTEVIZ_CACHE_DIR", "cache"))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.openai_api_key:
            print("Warning: OPENAI_API_KEY not set")
            return False
        return True

# Global configuration instance
config = Config() 
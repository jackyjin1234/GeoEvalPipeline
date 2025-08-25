"""
Configuration reader utility for the visualization system.
Reads pipeline configuration to adapt visualizations to different experimental setups.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

class ConfigReader:
    """Reads and parses pipeline configuration for visualization adaptation."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize config reader.
        
        Args:
            config_path: Path to pipeline config file. If None, uses default location.
        """
        if config_path is None:
            config_path = "/scratch/kesuniot_root/kesuniot0/jackyjin/GeoEvalPipeline/config/pipeline_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if loading fails."""
        return {
            'dataset': {
                'cities': ['aachen', 'bochum', 'bremen', 'cologne'],
                'images_per_city': 3
            },
            'features': {
                'target_features': ['human', 'object', 'vehicle', 'construction']
            },
            'output': {
                'base_directory': 'pipeline/output'
            }
        }
    
    @property
    def cities(self) -> List[str]:
        """Get list of cities from config."""
        return self.config.get('dataset', {}).get('cities', ['aachen', 'bochum', 'bremen', 'cologne'])
    
    @property
    def images_per_city(self) -> int:
        """Get number of images per city from config."""
        return self.config.get('dataset', {}).get('images_per_city', 3)
    
    @property
    def target_features(self) -> List[str]:
        """Get target features from config."""
        return self.config.get('features', {}).get('target_features', ['human', 'object', 'vehicle', 'construction'])
    
    @property
    def output_base_directory(self) -> str:
        """Get output base directory from config."""
        return self.config.get('output', {}).get('base_directory', 'pipeline/output')
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get output paths for different pipeline components."""
        base_path = Path("/scratch/kesuniot_root/kesuniot0/jackyjin/GeoEvalPipeline") / self.output_base_directory
        
        return {
            'evaluation_results': base_path / 'evaluation_results' / 'evaluation_results.json',
            'clipaway_results': base_path / 'clipaway_results',
            'masks': base_path / 'masks',
            'processed_original': base_path / 'processed' / 'original'
        }
    
    def get_expected_image_count(self) -> int:
        """Calculate expected total number of images based on config."""
        return len(self.cities) * self.images_per_city
    
    def print_config_summary(self):
        """Print a summary of the loaded configuration."""
        print("=== Configuration Summary ===")
        print(f"Cities: {', '.join(self.cities)} ({len(self.cities)} cities)")
        print(f"Images per city: {self.images_per_city}")
        print(f"Total expected images: {self.get_expected_image_count()}")
        print(f"Target features: {', '.join(self.target_features)} ({len(self.target_features)} features)")
        print(f"Output directory: {self.output_base_directory}")
        print("=" * 30)
#!/usr/bin/env python3
"""
Pipeline Configuration Management

This module handles loading, validation, and management of pipeline configuration
with support for YAML configuration files and environment variable overrides.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset-related configuration"""
    cityscapes_root: str
    images_per_city: int
    cities: List[str]
    selection_method: str = "systematic"
    
    def __post_init__(self):
        self.cityscapes_root = Path(self.cityscapes_root).expanduser().resolve()


@dataclass 
class FeaturesConfig:
    """Visual features configuration"""
    target_features: List[str]
    combine_masks: bool = False
    mask_suffix: str = "mask"


@dataclass
class ProcessingConfig:
    """Image processing configuration"""
    image_size: int = 512
    downscale_interpolation: str = "area"
    upscale_interpolation: str = "cubic"
    batch_size: int = 5


@dataclass
class CLIPAwayConfig:
    """CLIPAway model configuration"""
    path: str = 'pipeline/components/CLIPAway'
    device: str = "cuda"
    strength: float = 1.0
    scale: int = 1
    seed: int = 42
    display_focused_embeds: bool = False
    model_key: str = "botp/stable-diffusion-v1-5-inpainting"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    model: str = "gpt-4-vision-preview"
    concurrent_requests: int = 3
    cache_responses: bool = True
    use_ground_truth: bool = True
    max_images: Optional[int] = None


@dataclass
class OutputConfig:
    """Output configuration"""
    base_directory: str
    structure: Dict[str, str] = field(default_factory=lambda: {
        "masks": "masks",
        "processed": "processed",
        "clipaway": "clipaway_results",
        "evaluation": "evaluation_results"
    })
    cleanup_intermediate: bool = False
    save_debug_info: bool = True
    
    def __post_init__(self):
        self.base_directory = Path(self.base_directory).expanduser().resolve()


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file: Optional[str] = None
    console: bool = True
    
    def __post_init__(self):
        if self.file:
            self.file = Path(self.file).expanduser().resolve()


@dataclass
class PerformanceConfig:
    """Performance and resource configuration"""
    max_workers: int = 4
    memory_limit_gb: int = 16
    gpu_memory_fraction: float = 0.8


@dataclass
class ResumeConfig:
    """Resume/checkpoint configuration"""
    enabled: bool = True
    checkpoint_frequency: int = 5
    state_file: str = "pipeline_state.json"
    
    def __post_init__(self):
        self.state_file = Path(self.state_file).expanduser().resolve()


class PipelineConfig:
    """
    Main pipeline configuration manager with support for YAML files,
    environment variable overrides, and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from file or defaults.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.raw_config = {}
        
        # Load configuration
        if self.config_path and self.config_path.exists():
            self._load_from_file()
        else:
            logger.info("Using default configuration")
            self.raw_config = self._get_default_config()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Create typed configuration objects
        self._create_config_objects()
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.raw_config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from: {self.config_path}")
        except (yaml.YAMLError, IOError) as e:
            raise ValueError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'dataset': {
                'cityscapes_root': '/home/jackyjin/cityscapesScripts',
                'images_per_city': 5,
                'cities': ['aachen', 'bochum', 'bremen'],
                'selection_method': 'systematic'
            },
            'features': {
                'target_features': ['human', 'vehicle', 'construction', 'nature'],
                'combine_masks': False,
                'mask_suffix': 'mask'
            },
            'processing': {
                'image_size': 512,
                'downscale_interpolation': 'area',
                'upscale_interpolation': 'cubic',
                'batch_size': 5
            },
            'clipaway': {
                'path': 'pipeline/components/CLIPAway',
                'device': 'cuda',
                'strength': 1.0,
                'scale': 1,
                'seed': 42,
                'display_focused_embeds': False,
                'model_key': 'botp/stable-diffusion-v1-5-inpainting'
            },
            'evaluation': {
                'model': 'gpt-4-vision-preview',
                'concurrent_requests': 3,
                'cache_responses': True,
                'use_ground_truth': True,
                'max_images': None
            },
            'output': {
                'base_directory': 'pipeline/output',
                'structure': {
                    'masks': 'masks',
                    'processed': 'processed',
                    'clipaway': 'clipaway_results',
                    'evaluation': 'evaluation_results'
                },
                'cleanup_intermediate': False,
                'save_debug_info': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'pipeline/logs/pipeline.log',
                'console': True
            },
            'performance': {
                'max_workers': 4,
                'memory_limit_gb': 16,
                'gpu_memory_fraction': 0.8
            },
            'resume': {
                'enabled': True,
                'checkpoint_frequency': 5,
                'state_file': 'pipeline/output/pipeline_state.json'
            }
        }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Environment variable mapping
        env_mappings = {
            'CITYSCAPES_ROOT': ['dataset', 'cityscapes_root'],
            'PIPELINE_IMAGES_PER_CITY': ['dataset', 'images_per_city'],
            'PIPELINE_CITIES': ['dataset', 'cities'],
            'PIPELINE_FEATURES': ['features', 'target_features'],
            'PIPELINE_DEVICE': ['clipaway', 'device'],
            'OPENAI_MODEL': ['evaluation', 'model'],
            'PIPELINE_OUTPUT_DIR': ['output', 'base_directory'],
            'PIPELINE_LOG_LEVEL': ['logging', 'level']
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Parse list values
                if env_var in ['PIPELINE_CITIES', 'PIPELINE_FEATURES']:
                    value = [x.strip() for x in value.split(',')]
                elif env_var == 'PIPELINE_IMAGES_PER_CITY':
                    value = int(value)
                
                # Set nested configuration value
                config_dict = self.raw_config
                for key in config_path[:-1]:
                    config_dict = config_dict.setdefault(key, {})
                config_dict[config_path[-1]] = value
                
                logger.info(f"Applied environment override: {env_var}={value}")
    
    def _create_config_objects(self):
        """Create typed configuration objects from raw config"""
        try:
            self.dataset = DatasetConfig(**self.raw_config.get('dataset', {}))
            self.features = FeaturesConfig(**self.raw_config.get('features', {}))
            self.processing = ProcessingConfig(**self.raw_config.get('processing', {}))
            self.clipaway = CLIPAwayConfig(**self.raw_config.get('clipaway', {}))
            self.evaluation = EvaluationConfig(**self.raw_config.get('evaluation', {}))
            self.output = OutputConfig(**self.raw_config.get('output', {}))
            self.logging = LoggingConfig(**self.raw_config.get('logging', {}))
            self.performance = PerformanceConfig(**self.raw_config.get('performance', {}))
            self.resume = ResumeConfig(**self.raw_config.get('resume', {}))
        except TypeError as e:
            raise ValueError(f"Configuration validation error: {e}")
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate dataset configuration
        if not self.dataset.cityscapes_root.exists():
            errors.append(f"Cityscapes root directory not found: {self.dataset.cityscapes_root}")
        
        if self.dataset.images_per_city <= 0:
            errors.append("images_per_city must be positive")
        
        if not self.dataset.cities:
            errors.append("At least one city must be specified")
        
        if self.dataset.selection_method not in ['systematic', 'first_n', 'evenly_spaced']:
            errors.append(f"Invalid selection method: {self.dataset.selection_method}")
        
        # Validate features configuration
        # if not self.features.target_features:
        #     errors.append("At least one target feature must be specified")
        
        # Validate processing configuration
        if self.processing.image_size <= 0:
            errors.append("image_size must be positive")
        
        if self.processing.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        # Validate CLIPAway configuration
        if self.clipaway.device not in ['cuda', 'cpu']:
            errors.append(f"Invalid device: {self.clipaway.device}")
        
        if not 0 <= self.clipaway.strength <= 2:
            errors.append("strength must be between 0 and 2")
        
        # Validate evaluation configuration
        if self.evaluation.concurrent_requests <= 0:
            errors.append("concurrent_requests must be positive")
        
        # Validate performance configuration
        if self.performance.max_workers <= 0:
            errors.append("max_workers must be positive")
        
        if self.performance.memory_limit_gb <= 0:
            errors.append("memory_limit_gb must be positive")
        
        if not 0 < self.performance.gpu_memory_fraction <= 1:
            errors.append("gpu_memory_fraction must be between 0 and 1")
        
        # Validate resume configuration
        if self.resume.checkpoint_frequency <= 0:
            errors.append("checkpoint_frequency must be positive")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        logger.info("Configuration validation passed")
    
    def create_output_directories(self):
        """Create output directory structure"""
        directories = [
            self.output.base_directory,
            self.output.base_directory / self.output.structure['masks'],
            self.output.base_directory / self.output.structure['processed'],
            self.output.base_directory / self.output.structure['clipaway'],
            self.output.base_directory / self.output.structure['evaluation']
        ]
        
        # Create log directory if specified
        if self.logging.file:
            directories.append(self.logging.file.parent)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories created at: {self.output.base_directory}")
    
    def get_masks_dir(self) -> Path:
        """Get masks output directory"""
        return self.output.base_directory / self.output.structure['masks']
    
    def get_processed_dir(self) -> Path:
        """Get processed images output directory"""
        return self.output.base_directory / self.output.structure['processed']
    
    def get_clipaway_dir(self) -> Path:
        """Get CLIPAway results output directory"""
        return self.output.base_directory / self.output.structure['clipaway']
    
    def get_evaluation_dir(self) -> Path:
        """Get evaluation results output directory"""
        return self.output.base_directory / self.output.structure['evaluation']
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file"""
        if not output_path:
            output_path = self.output.base_directory / "pipeline_config_used.yaml"
        
        config_to_save = deepcopy(self.raw_config)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return deepcopy(self.raw_config)
    
    def __repr__(self) -> str:
        return f"PipelineConfig(config_path={self.config_path})"


if __name__ == "__main__":
    # Test configuration loading
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        config = PipelineConfig(config_path)
        print("Configuration loaded successfully:")
        print(f"  Dataset root: {config.dataset.cityscapes_root}")
        print(f"  Cities: {config.dataset.cities}")
        print(f"  Images per city: {config.dataset.images_per_city}")
        print(f"  Target features: {config.features.target_features}")
        print(f"  Output directory: {config.output.base_directory}")
        
        # Test directory creation
        config.create_output_directories()
        print("Output directories created successfully")
        
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
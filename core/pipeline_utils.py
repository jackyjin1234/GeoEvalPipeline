#!/usr/bin/env python3
"""
Pipeline Utilities

Common utility functions and helpers for the visual cue evaluation pipeline.
"""

import logging
import os
import psutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import torch


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup logging configuration for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        console: Whether to log to console
    
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler if requested
    # if console:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Get pipeline-specific logger
    logger = logging.getLogger('pipeline')
    logger.info(f"Logging configured - Level: {level}, File: {log_file}")
    
    return logger


def validate_environment() -> Dict[str, Any]:
    """
    Validate system environment and dependencies for pipeline execution.
    
    Returns:
        Dictionary with validation results and system information
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'system_info': {},
        'dependencies': {}
    }
    
    # Check Python version
    python_version = sys.version_info
    validation_results['system_info']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    
    if python_version < (3, 8):
        validation_results['errors'].append("Python 3.8 or higher required")
        validation_results['valid'] = False
    
    # Check system resources
    memory_gb = psutil.virtual_memory().total / (1024**3)
    validation_results['system_info']['memory_gb'] = round(memory_gb, 1)
    
    if memory_gb < 8:
        validation_results['warnings'].append(f"Low system memory: {memory_gb:.1f}GB (recommended: 16GB+)")
    
    disk_usage = psutil.disk_usage('/')
    free_gb = disk_usage.free / (1024**3)
    validation_results['system_info']['free_disk_gb'] = round(free_gb, 1)
    
    if free_gb < 10:
        validation_results['warnings'].append(f"Low disk space: {free_gb:.1f}GB (recommended: 50GB+)")
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    validation_results['system_info']['gpu_available'] = gpu_available
    
    if gpu_available:
        validation_results['system_info']['gpu_count'] = torch.cuda.device_count()
        validation_results['system_info']['gpu_name'] = torch.cuda.get_device_name(0)
        
        # Check GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        validation_results['system_info']['gpu_memory_gb'] = round(gpu_memory_gb, 1)
        
        if gpu_memory_gb < 6:
            validation_results['warnings'].append(f"Low GPU memory: {gpu_memory_gb:.1f}GB (recommended: 8GB+)")
    else:
        validation_results['warnings'].append("No GPU detected - CLIPAway processing will be slow")
    
    # Check required Python packages
    required_packages = [
        'torch', 'torchvision', 'PIL', 'cv2', 'numpy', 'yaml', 
        'diffusers', 'aiohttp', 'psutil'
    ]
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                validation_results['dependencies'][package] = cv2.__version__
            elif package == 'PIL':
                from PIL import Image
                validation_results['dependencies'][package] = Image.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                validation_results['dependencies'][package] = version
        except ImportError:
            validation_results['errors'].append(f"Required package not found: {package}")
            validation_results['valid'] = False
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        validation_results['warnings'].append("OPENAI_API_KEY environment variable not set")
    
    return validation_results


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_processing_time(num_items: int, phases: int = 7) -> Dict[str, float]:
    """
    Estimate processing time based on number of items and system capabilities.
    
    Args:
        num_items: Number of images to process
        phases: Number of pipeline phases
    
    Returns:
        Dictionary with time estimates in seconds
    """
    # Base time estimates per item per phase (in seconds)
    base_times = {
        'mask_generation': 2.0,
        'image_processing': 1.0,
        'clipaway_processing': 15.0,  # Most time-consuming
        'evaluation': 10.0,  # API calls
        'other_phases': 0.5
    }
    
    # Adjust for system capabilities
    gpu_available = torch.cuda.is_available()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # GPU acceleration factor
    gpu_factor = 0.3 if gpu_available else 1.0
    
    # Memory factor (faster with more RAM)
    memory_factor = min(1.0, 8.0 / memory_gb) if memory_gb > 4 else 1.5
    
    estimates = {
        'mask_generation': num_items * base_times['mask_generation'] * memory_factor,
        'image_processing': num_items * base_times['image_processing'] * memory_factor,
        'clipaway_processing': num_items * base_times['clipaway_processing'] * gpu_factor,
        'evaluation': num_items * base_times['evaluation'],  # Network bound
        'other_phases': num_items * base_times['other_phases'] * (phases - 4)
    }
    
    estimates['total'] = sum(estimates.values())
    
    return estimates


def monitor_system_resources() -> Dict[str, Any]:
    """Monitor current system resource usage"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
        'gpu_memory_used': torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
        'gpu_memory_percent': (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100) 
                             if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 0 else 0
    }


def create_progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a text-based progress bar"""
    if total == 0:
        return "[" + " " * width + "] 0%"
    
    progress = current / total
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    percentage = progress * 100
    
    return f"[{bar}] {percentage:.1f}% ({current}/{total})"


def safe_file_operation(operation: callable, *args, max_retries: int = 3, 
                       delay: float = 1.0, **kwargs) -> Any:
    """
    Safely execute file operations with retry logic.
    
    Args:
        operation: Function to execute
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        *args, **kwargs: Arguments for the operation
    
    Returns:
        Result of the operation
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except (OSError, IOError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
                continue
            break
    
    raise last_exception


def validate_file_integrity(file_path: Path, expected_size: Optional[int] = None,
                          check_readable: bool = True) -> bool:
    """
    Validate file integrity and accessibility.
    
    Args:
        file_path: Path to file to validate
        expected_size: Expected file size in bytes (optional)
        check_readable: Whether to check if file is readable
    
    Returns:
        True if file is valid, False otherwise
    """
    try:
        if not file_path.exists():
            return False
        
        if not file_path.is_file():
            return False
        
        file_size = file_path.stat().st_size
        
        if expected_size is not None and file_size != expected_size:
            return False
        
        if check_readable:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        
        return True
    
    except (OSError, IOError, PermissionError):
        return False


def cleanup_directory(directory: Path, pattern: str = "*", 
                     exclude_patterns: Optional[List[str]] = None,
                     dry_run: bool = False) -> Tuple[int, int]:
    """
    Clean up files in directory matching pattern.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        exclude_patterns: Patterns to exclude from deletion
        dry_run: If True, only simulate cleanup
    
    Returns:
        Tuple of (files_deleted, bytes_freed)
    """
    exclude_patterns = exclude_patterns or []
    files_deleted = 0
    bytes_freed = 0
    
    if not directory.exists():
        return files_deleted, bytes_freed
    
    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue
        
        # Check exclusion patterns
        excluded = False
        for exclude_pattern in exclude_patterns:
            if file_path.match(exclude_pattern):
                excluded = True
                break
        
        if excluded:
            continue
        
        try:
            file_size = file_path.stat().st_size
            
            if not dry_run:
                file_path.unlink()
            
            files_deleted += 1
            bytes_freed += file_size
            
        except (OSError, IOError) as e:
            logging.getLogger(__name__).warning(f"Failed to delete {file_path}: {e}")
    
    return files_deleted, bytes_freed


def get_pipeline_version() -> str:
    """Get pipeline version information"""
    try:
        import pipeline
        return pipeline.__version__
    except ImportError:
        return "1.0.0"


def generate_run_summary(stats: Dict[str, Any], duration: float) -> str:
    """Generate a human-readable run summary"""
    summary_lines = [
        "=" * 60,
        "PIPELINE EXECUTION SUMMARY",
        "=" * 60,
        f"Duration: {format_duration(duration)}",
        f"Items processed: {stats.get('total_items', 0)}",
        f"Items completed: {stats.get('items_completed', 0)}",
        f"Items failed: {stats.get('items_failed', 0)}",
        f"Phases completed: {stats.get('phases_completed', 0)}/{stats.get('total_phases', 0)}",
        f"Checkpoints saved: {stats.get('checkpoints_saved', 0)}",
        ""
    ]
    
    success_rate = 0
    if stats.get('total_items', 0) > 0:
        success_rate = (stats.get('items_completed', 0) / stats.get('total_items', 0)) * 100
    
    summary_lines.append(f"Success rate: {success_rate:.1f}%")
    
    if stats.get('items_failed', 0) > 0:
        summary_lines.append(f"⚠️  {stats.get('items_failed', 0)} items failed processing")
    
    summary_lines.extend([
        "",
        f"Pipeline version: {get_pipeline_version()}",
        f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60
    ])
    
    return "\n".join(summary_lines)


if __name__ == "__main__":
    # Test utility functions
    print("Testing pipeline utilities...")
    
    # Test logging setup
    logger = setup_logging("DEBUG")
    logger.info("Logging test successful")
    
    # Test environment validation
    validation = validate_environment()
    print(f"\nEnvironment validation: {'✓' if validation['valid'] else '✗'}")
    print(f"Errors: {len(validation['errors'])}")
    print(f"Warnings: {len(validation['warnings'])}")
    
    # Test time estimation
    estimates = estimate_processing_time(15, 7)
    print(f"\nEstimated processing time for 15 items:")
    for phase, time_est in estimates.items():
        print(f"  {phase}: {format_duration(time_est)}")
    
    # Test system monitoring
    resources = monitor_system_resources()
    print(f"\nSystem resources:")
    print(f"  CPU: {resources['cpu_percent']:.1f}%")
    print(f"  Memory: {resources['memory_percent']:.1f}% ({resources['memory_used_gb']:.1f}GB)")
    
    # Test progress bar
    print(f"\nProgress bar examples:")
    for i in [0, 25, 50, 75, 100]:
        print(f"  {create_progress_bar(i, 100)}")
    
    print("\nUtilities test completed successfully!")
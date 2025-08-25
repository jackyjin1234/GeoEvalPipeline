#!/usr/bin/env python3
"""
Visual Cue Evaluation Pipeline CLI

Command-line interface for running the visual cue evaluation pipeline
with support for configuration, validation, resume, and progress monitoring.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from core.pipeline_config import PipelineConfig
from core.visual_cue_pipeline import VisualCuePipeline
from core.pipeline_utils import setup_logging, validate_environment, format_duration


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Visual Cue Evaluation Pipeline - Evaluate importance of visual cues for geolocation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    # Run with custom configuration
    python run_pipeline.py --config config/pipeline_config.yaml
    """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/pipeline_config.yaml',
        help='Configuration file path (default: config/pipeline_config.yaml)'
    )
    
    # Execution options
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate setup without running pipeline'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without actual processing'
    )
    
    
    
    return parser.parse_args()




def setup_pipeline_logging(config: PipelineConfig):
    """Setup logging based on configuration"""
    log_level = config.logging.level
    log_file = config.logging.file
    
    logger = setup_logging(log_level, log_file)
    
    return logger


def print_pipeline_info(config: PipelineConfig):
    """Print pipeline configuration information"""
    print("=" * 60)
    print("VISUAL CUE EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Dataset root: {config.dataset.cityscapes_root}")
    print(f"Cities: {', '.join(config.dataset.cities)}")
    print(f"Images per city: {config.dataset.images_per_city}")
    if config.features.target_features:
        print(f"Target features: {', '.join(config.features.target_features)}")
    print(f"Device: {config.clipaway.device}")
    print(f"Evaluation model: {config.evaluation.model}")
    print(f"Output directory: {config.output.base_directory}")
    print("=" * 60)


def print_validation_results(validation: dict):
    """Print validation results"""
    if validation['valid']:
        print("✓ Pipeline validation passed")
    else:
        print("✗ Pipeline validation failed")
        print("\nErrors:")
        for error in validation['errors']:
            print(f"  • {error}")
    
    if validation['warnings']:
        print("\nWarnings:")
        for warning in validation['warnings']:
            print(f"  ⚠ {warning}")


async def run_pipeline_async(config: PipelineConfig, args: argparse.Namespace) -> bool:
    """Run the pipeline asynchronously"""
    try:
        load_dotenv()
        # Create pipeline
        pipeline = VisualCuePipeline(config)
        
        # Validate setup
        validation = pipeline.validate_setup()
        print_validation_results(validation)
        
        if not validation['valid']:
            return False
        
        if getattr(args, 'validate_only', False):
            print("\nValidation complete - exiting (--validate-only)")
            return True
        
        if getattr(args, 'dry_run', False):
            print("\nDry run complete - no actual processing performed")
            return True
        
        # Print pipeline info
        print_pipeline_info(config)
        
        # Run pipeline
        start_time = asyncio.get_event_loop().time()
        
        results = await pipeline.run_pipeline()
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # Print results
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETED")
        print("=" * 60)
        print(f"Items processed: {results['total_items_processed']}")
        print(f"Success rate: {results['success_rate']:.1f}%")
        print(f"Execution time: {format_duration(duration)}")
        print(f"Output directory: {results['output_directory']}")
        print(f"Summary: {results['summary']}")
        print("=" * 60)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return False
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Pipeline execution failed: {e}", exc_info=True)
        print(f"\nPipeline execution failed: {e}")
        return False


def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            print("Use --config to specify a different configuration file")
            sys.exit(1)
        
        print(f"Loading configuration from: {config_path}")
        config = PipelineConfig(str(config_path))
        
        # Setup logging
        logger = setup_pipeline_logging(config)
        logger.info(f"Pipeline CLI started with config: {config_path}")
        
        # Apply command line overrides
        # apply_argument_overrides(config, args)
        
        # Create output directories
        config.create_output_directories()
        
        # Run pipeline
        success = asyncio.run(run_pipeline_async(config, args))
        
        if success:
            print("\nPipeline completed successfully!")
            sys.exit(0)
        else:
            print("\nPipeline failed or was interrupted")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.getLogger(__name__).error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
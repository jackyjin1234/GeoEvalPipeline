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

# Add parent directories to path for external dependencies and relative imports
import sys
pipeline_root = Path(__file__).parent
cityscapes_root = pipeline_root.parent
sys.path.insert(0, str(pipeline_root))  # For direct imports (core, adapters, utils)
sys.path.insert(0, str(cityscapes_root))  # For external dependencies (cityscapesscripts, CLIPAway)

from core.pipeline_config import PipelineConfig
from core.visual_cue_pipeline import VisualCuePipeline
from core.pipeline_utils import setup_logging, validate_environment, format_duration


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Visual Cue Evaluation Pipeline - Evaluate importance of visual cues for geolocation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python run_pipeline.py
  
  # Run with custom configuration
  python run_pipeline.py --config config/pipeline_config.yaml
  
  # Run with test configuration
  python run_pipeline.py --config config/test_config.yaml
  
  # Resume interrupted pipeline
  python run_pipeline.py --config config/pipeline_config.yaml --resume
  
  # Validate setup without running
  python run_pipeline.py --validate-only
  
  # Run with specific features only
  python run_pipeline.py --config config/pipeline_config.yaml --features "human,vehicle"
  
  # Limit number of images for testing
  python run_pipeline.py --config config/test_config.yaml --max-images 5
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
        '--resume', '-r',
        action='store_true',
        help='Resume from previous execution state'
    )
    
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
    
    # Pipeline overrides
    parser.add_argument(
        '--features',
        type=str,
        help='Comma-separated list of features to evaluate (overrides config)'
    )
    
    parser.add_argument(
        '--cities',
        type=str,
        help='Comma-separated list of cities to process (overrides config)'
    )
    
    parser.add_argument(
        '--images-per-city',
        type=int,
        help='Number of images per city (overrides config)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        help='Maximum total images to process (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='Device to use for processing (overrides config)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: use config or console only)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress console output (log to file only)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up intermediate files after completion'
    )
    
    # Development/testing options
    parser.add_argument(
        '--mock-evaluation',
        action='store_true',
        help='Use mock evaluation API for testing'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    return parser.parse_args()


def apply_argument_overrides(config: PipelineConfig, args: argparse.Namespace):
    """Apply command line argument overrides to configuration"""
    # Features override
    if args.features:
        config.features.target_features = [f.strip() for f in args.features.split(',')]
        print(f"Override: features = {config.features.target_features}")
    
    # Cities override
    if args.cities:
        config.dataset.cities = [c.strip() for c in args.cities.split(',')]
        print(f"Override: cities = {config.dataset.cities}")
    
    # Images per city override
    if args.images_per_city:
        config.dataset.images_per_city = args.images_per_city
        print(f"Override: images_per_city = {config.dataset.images_per_city}")
    
    # Max images override
    if args.max_images:
        config.evaluation.max_images = args.max_images
        print(f"Override: max_images = {config.evaluation.max_images}")
    
    # Device override
    if args.device:
        config.clipaway.device = args.device
        print(f"Override: device = {config.clipaway.device}")
    
    # Output directory override
    if args.output_dir:
        config.output.base_directory = Path(args.output_dir)
        print(f"Override: output_dir = {config.output.base_directory}")
    
    # Mock evaluation override
    if args.mock_evaluation:
        config.evaluation.model = "mock"
        print("Override: using mock evaluation API")
    
    # Cleanup override
    if args.cleanup:
        config.output.cleanup_intermediate = True
        print("Override: cleanup intermediate files enabled")


def setup_pipeline_logging(args: argparse.Namespace, config: Optional[PipelineConfig] = None):
    """Setup logging based on arguments and configuration"""
    log_level = args.log_level
    
    # Determine log file
    log_file = None
    if args.log_file:
        log_file = Path(args.log_file)
    elif config and config.logging.file:
        log_file = config.logging.file
    
    # Setup logging
    console_logging = not args.quiet
    logger = setup_logging(log_level, log_file, console_logging)
    
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
        
        if args.validate_only:
            print("\nValidation complete - exiting (--validate-only)")
            return True
        
        if args.dry_run:
            print("\nDry run complete - no actual processing performed")
            return True
        
        # Print pipeline info
        print_pipeline_info(config)
        
        # Run pipeline
        print(f"\nStarting pipeline execution (resume: {args.resume})...")
        start_time = asyncio.get_event_loop().time()
        
        results = await pipeline.run_pipeline(resume=args.resume)
        
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
        print("Pipeline state has been saved - use --resume to continue")
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
        logger = setup_pipeline_logging(args, config)
        logger.info(f"Pipeline CLI started with config: {config_path}")
        
        # Apply command line overrides
        apply_argument_overrides(config, args)
        
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
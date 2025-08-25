# Visual Cue Evaluation Pipeline

An integrated pipeline for evaluating the importance of visual cues in geolocation tasks using the Cityscapes dataset. This pipeline orchestrates binary mask generation, image processing, visual cue removal using CLIPAway, and geolocation evaluation to systematically assess how removing specific visual features affects GPT's geolocation accuracy.

## Overview

The pipeline consists of 7 main phases:

1. **Dataset Discovery** - Find and validate image/annotation/GPS triplets
2. **Image Selection** - Systematically select images for processing  
3. **Mask Generation** - Create binary masks for visual features
4. **Image Processing** - Split and downscale images to 512x512
5. **CLIPAway Processing** - Remove visual cues using AI inpainting
6. **Evaluation** - Test geolocation accuracy with GPT vision models
7. **Report Generation** - Generate comprehensive analysis reports

## Key Features

- **Optimized Data Flow**: Eliminates redundant file operations through ordered processing
- **Resume Capability**: Handle interruptions and continue from last checkpoint
- **Systematic Selection**: Deterministic image selection for reproducible results
- **Ground Truth Integration**: Uses actual GPS coordinates from Cityscapes vehicle data
- **Comprehensive Testing**: Full test suite with mock adapters for development
- **Flexible Configuration**: YAML-based configuration with command-line overrides
- **Performance Monitoring**: Resource usage tracking and time estimation

## Installation

### Prerequisites

- Python 3.8+
- Required packages: `torch`, `torchvision`, `PIL`, `cv2`, `numpy`, `yaml`, `aiohttp`
- Optional: GPU with CUDA support for faster CLIPAway processing
- OpenAI API key for geolocation evaluation

### Setup

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision Pillow opencv-python numpy PyYAML aiohttp psutil
   pip install diffusers  # For CLIPAway
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export CITYSCAPES_DATASET="/path/to/cityscapes"
   ```

3. **Download CLIPAway Models** (if using actual CLIPAway):
   ```bash
   cd CLIPAway
   bash download_pretrained_models.sh
   ```

## Configuration

The pipeline uses YAML configuration files. See `config/pipeline_config.yaml` for the main configuration and `config/test_config.yaml` for testing.

### Key Configuration Sections:

```yaml
dataset:
  cityscapes_root: "/path/to/cityscapes"
  images_per_city: 5
  cities: ["aachen", "bochum", "bremen"]
  
features:
  target_features: ["human", "vehicle", "construction", "nature"]
  
evaluation:
  model: "gpt-4-vision-preview"
  concurrent_requests: 3
  use_ground_truth: true
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python run_pipeline.py

# Run with custom configuration
python run_pipeline.py --config config/pipeline_config.yaml

# Resume interrupted execution
python run_pipeline.py --config config/pipeline_config.yaml --resume

# Test run with limited images
python run_pipeline.py --config config/test_config.yaml --max-images 5
```

### Command Line Options

```bash
# Configuration
--config PATH              Configuration file path
--features LIST            Override target features (comma-separated)
--cities LIST             Override cities to process
--max-images N            Limit total images processed

# Execution
--resume                  Resume from previous state
--validate-only           Only validate setup
--dry-run                 Simulate execution without processing

# Output
--output-dir PATH         Override output directory
--log-level LEVEL         Set logging level (DEBUG, INFO, WARNING, ERROR)
--quiet                   Suppress console output
```

### Examples

```bash
# Full evaluation with all default features
python run_pipeline.py --config config/pipeline_config.yaml --features "human,vehicle,construction,nature"

# Quick test with mock evaluation
python run_pipeline.py --config config/test_config.yaml --mock-evaluation --max-images 3

# Process specific cities only
python run_pipeline.py --cities "aachen,bremen" --images-per-city 3

# Validate setup without running
python run_pipeline.py --validate-only
```

## Architecture

### Core Components

- **`core/visual_cue_pipeline.py`** - Main pipeline orchestrator
- **`core/pipeline_dataset.py`** - Dataset discovery and management
- **`core/pipeline_config.py`** - Configuration system
- **`core/pipeline_state.py`** - State management and resume capability
- **`core/pipeline_utils.py`** - Utility functions and helpers

### Adapters

- **`adapters/mask_generator.py`** - Binary mask generation (wraps `createBinaryMasks.py`)
- **`adapters/image_processor.py`** - Image processing (wraps `image_processor.py`)
- **`adapters/clipaway_adapter.py`** - CLIPAway integration
- **`adapters/evaluator_adapter.py`** - Geolocation evaluation

### Data Flow

```
Original Images + Annotations + GPS Data
    ↓
Dataset Discovery & Selection
    ↓
Binary Mask Generation (per feature)
    ↓
Image Processing (split to 512x512)
    ↓
CLIPAway Processing (remove visual cues)
    ↓
Geolocation Evaluation (compare accuracy)
    ↓
Results Analysis & Reporting
```

## Testing

The pipeline includes a comprehensive test suite:

```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test categories
python tests/run_all_tests.py --category dataset
python tests/run_all_tests.py --category adapters
python tests/run_all_tests.py --category pipeline

# Run with verbose output
python tests/run_all_tests.py --verbose

# Fast testing (skip slow integration tests)
python tests/run_all_tests.py --fast
```

### Test Categories

- **Pipeline Tests** - End-to-end integration tests
- **Dataset Tests** - Dataset discovery and selection validation
- **Adapter Tests** - Component integration tests
- **Component Tests** - Individual component functionality

## Output Structure

```
pipeline/output/
├── masks/                    # Generated binary masks
│   ├── human/
│   ├── vehicle/
│   └── ...
├── processed/               # Split and downscaled images
│   ├── original/
│   ├── human/
│   └── ...
├── clipaway_results/        # Images with visual cues removed
│   ├── human/
│   ├── vehicle/
│   └── ...
├── evaluation_results/      # Geolocation evaluation data
│   ├── evaluation_results.db
│   ├── evaluation_results.json
│   └── evaluation_summary.txt
├── pipeline_report.json    # Comprehensive execution report
├── pipeline_config_used.yaml
└── pipeline_state.json     # Resume state
```

## Performance Optimization

### Dataset Processing
- **Ordered Processing**: Files processed in predetermined order
- **Batch Operations**: Process multiple items simultaneously
- **Smart Caching**: Avoid redundant file operations
- **Memory Management**: Stream processing for large datasets

### GPU Utilization
- **Model Reuse**: CLIPAway models loaded once and reused
- **Memory Monitoring**: Track GPU memory usage
- **Batch Size Optimization**: Automatic batch sizing based on available memory

### API Efficiency
- **Concurrent Requests**: Multiple GPT API calls in parallel
- **Response Caching**: Cache geolocation predictions
- **Rate Limiting**: Respect API rate limits

## Resume Capability

The pipeline supports resuming interrupted executions:

1. **State Persistence**: Automatically saves progress after each phase
2. **Smart Recovery**: Identifies completed work and continues from interruption point
3. **Validation**: Verifies existing outputs before resuming
4. **Progress Reporting**: Shows detailed resume information

```bash
# Resume interrupted pipeline
python run_pipeline.py --resume

# View resume status
python run_pipeline.py --validate-only  # Shows current state
```

## Error Handling

- **Graceful Degradation**: Continue processing when individual items fail
- **Comprehensive Logging**: Detailed error reporting and debugging info
- **Validation Gates**: Validate inputs before each processing phase
- **Recovery Strategies**: Multiple fallback options for common failures

## Development

### Adding New Features

1. **Visual Features**: Add to `config/pipeline_config.yaml` `target_features` list
2. **Selection Methods**: Extend `PipelineDataset.select_images_per_city()`
3. **Evaluation Models**: Add support in `EvaluatorAdapter`
4. **New Phases**: Add to `PipelinePhase` enum and implement in orchestrator

### Testing New Components

1. Create test class in appropriate `tests/test_*.py` file
2. Add mock data generation methods
3. Test both success and failure cases
4. Include in `tests/run_all_tests.py`

### Configuration Extensions

1. Add new section to configuration dataclasses in `pipeline_config.py`
2. Update validation methods
3. Add command-line overrides if needed
4. Update documentation

## Troubleshooting

### Common Issues

**"No valid items selected"**
- Check dataset paths in configuration
- Verify image/annotation/GPS file alignment
- Ensure GPS files contain valid coordinates

**"CLIPAway model loading failed"**
- Download required models: `bash CLIPAway/download_pretrained_models.sh`
- Check GPU memory availability
- Try CPU mode: `--device cpu`

**"API rate limit exceeded"**
- Reduce `concurrent_requests` in configuration
- Enable response caching: `cache_responses: true`
- Use mock mode for testing: `--mock-evaluation`

**"Pipeline validation failed"**
- Run `--validate-only` to see specific issues
- Check Python version (3.8+ required)
- Verify all dependencies are installed

### Performance Issues

**Slow processing**:
- Enable GPU processing for CLIPAway
- Increase batch sizes if memory allows
- Use SSD storage for faster I/O
- Enable intermediate file cleanup

**Memory issues**:
- Reduce batch sizes
- Process fewer images per run
- Enable cleanup of intermediate files
- Monitor with `pipeline_utils.monitor_system_resources()`

## Contributing

1. **Code Style**: Follow existing patterns and naming conventions
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update README and docstrings
4. **Configuration**: Ensure new features are configurable

## License

This pipeline integrates with the Cityscapes dataset and existing tools. Please ensure compliance with their respective licenses and usage terms.

## Acknowledgments

- Cityscapes dataset team for the comprehensive urban scene dataset
- CLIPAway authors for the visual cue removal methodology
- OpenAI for the GPT vision models used in evaluation
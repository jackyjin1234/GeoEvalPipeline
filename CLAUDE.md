# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Visual Cue Evaluation Pipeline for assessing geolocation accuracy using the Cityscapes dataset. The pipeline orchestrates binary mask generation, image processing, visual cue removal using CLIPAway, and geolocation evaluation to systematically test how removing specific visual features affects GPT's geolocation accuracy.

## Development Commands

### Main Pipeline Execution
```bash
# Run pipeline with default configuration
python run_pipeline.py

# Run with custom configuration
python run_pipeline.py --config config/pipeline_config.yaml

# Test run with test configuration
python run_pipeline.py --config config/test_config.yaml

# Validate setup without running
python run_pipeline.py --validate-only

# Dry run without actual processing
python run_pipeline.py --dry-run
```

### Testing
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

### Environment Setup
```bash
# Install core dependencies
pip install torch torchvision Pillow opencv-python numpy PyYAML aiohttp psutil

# Install CLIPAway dependencies
pip install diffusers

# Download CLIPAway models
cd CLIPAway && bash download_pretrained_models.sh
```

## Architecture Overview

### Pipeline Phases
The pipeline executes in 7 main phases:
1. **Dataset Discovery** - Find image/annotation/GPS triplets (`core/pipeline_dataset.py`)
2. **Image Selection** - Systematic selection for reproducible results
3. **Mask Generation** - Create binary masks for visual features (`adapters/mask_generator.py`)
4. **Image Processing** - Split and downscale to 512x512 (`adapters/image_processor.py`)
5. **CLIPAway Processing** - Remove visual cues using AI inpainting (`adapters/clipaway_adapter.py`)
6. **Evaluation** - Test geolocation accuracy with GPT (`adapters/evaluator_adapter.py`)
7. **Report Generation** - Generate comprehensive analysis reports

### Core Components
- **`core/visual_cue_pipeline.py`** - Main pipeline orchestrator that coordinates all phases
- **`core/pipeline_dataset.py`** - Dataset discovery and management, handles Cityscapes structure
- **`core/pipeline_config.py`** - Configuration system with dataclasses and validation
- **`core/pipeline_state.py`** - State management for resume capability
- **`core/pipeline_utils.py`** - Utility functions and system monitoring

### Adapter Pattern
All external integrations use adapters:
- **`adapters/mask_generator.py`** - Wraps `utils/createBinaryMasks.py`
- **`adapters/image_processor.py`** - Wraps `utils/image_processor.py` 
- **`adapters/clipaway_adapter.py`** - Integrates with CLIPAway model
- **`adapters/evaluator_adapter.py`** - Handles GPT API calls for geolocation

### Data Flow
```
Cityscapes Dataset (images + annotations + GPS)
    ↓
Dataset Discovery & Selection
    ↓  
Binary Mask Generation (per feature: human, vehicle, etc.)
    ↓
Image Processing (split to 512x512 quadrants)
    ↓
CLIPAway Processing (remove visual cues using masks)
    ↓
Geolocation Evaluation (GPT-4 vision model)
    ↓
Results Analysis & Report Generation
```

## Configuration System

### Main Config Files
- **`config/pipeline_config.yaml`** - Production configuration
- **`config/test_config.yaml`** - Testing configuration with minimal processing

### Key Configuration Sections
```yaml
dataset:
  cityscapes_root: "/path/to/cityscapes"
  cities: ["aachen", "bochum", "bremen"]
  images_per_city: 5

features:
  target_features: ["human", "vehicle", "construction", "nature"]

evaluation:
  model: "gpt-4o"
  concurrent_requests: 3
  use_ground_truth: true
```

## Cityscapes Dataset Structure
The pipeline expects standard Cityscapes structure:
```
cityscapes_root/
├── leftImg8bit/train/[city]/[city]_[seq]_[frame]_leftImg8bit.png
├── gtFine/train/[city]/[city]_[seq]_[frame]_gtFine_polygons.json
└── vehicle/train/[city]/[city]_[seq]_[frame]_vehicle.json  # GPS data
```


## Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
export CITYSCAPES_DATASET="/path/to/cityscapes"
```

## Output Structure
```
pipeline/output/
├── masks/[feature]/           # Generated binary masks
├── processed/[feature]/       # Split images (512x512)
├── clipaway_results/[feature]/ # Inpainted images
├── evaluation_results/        # GPT evaluation results
└── pipeline_report.json      # Comprehensive execution report
```

## Common Issues

### Dataset Path Issues
- Ensure `cityscapes_root` points to directory containing `leftImg8bit/`, `gtFine/`, `vehicle/`
- GPS files must be present in `vehicle/` directory with valid coordinates

### CLIPAway Setup
- Download models: `cd CLIPAway && bash download_pretrained_models.sh`
- For GPU issues, try `device: "cpu"` in config
- Check GPU memory with `gpu_memory_fraction` setting

### API Rate Limits  
- Reduce `concurrent_requests` in evaluation config
- Enable `cache_responses: true` to avoid duplicate calls
- Use `--mock-evaluation` for testing without API calls

## Development Patterns

### Adding New Visual Features
1. Add to `features.target_features` in config
2. Ensure feature mapping exists in mask generation logic
3. Test with single city/image first

### Extending Evaluation Models
1. Add model support in `adapters/evaluator_adapter.py`
2. Update configuration schema in `core/pipeline_config.py`
3. Add corresponding test cases

### Mock Components for Testing
All adapters have mock modes for development:
- Set `mock: true` in adapter config sections
- Use `config/test_config.yaml` for rapid testing
- Mock evaluation prevents API usage during development
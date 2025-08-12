# Learning Exercises for Research Beginners

## Exercise Set 1: Understanding the Architecture

### Ex 1.1: Code Reading (30 minutes)
1. Open `core/visual_cue_pipeline.py`
2. Find the `run_pipeline()` method (line ~150)
3. Identify the 7 main phases
4. **Question**: Why are phases executed in this specific order?

### Ex 1.2: Configuration Understanding (20 minutes)
1. Open `config/pipeline_config.yaml`
2. Change `images_per_city` to 1
3. Run: `python run_pipeline.py --validate-only`
4. **Question**: What validation errors appear and why?

### Ex 1.3: Data Structure Exploration (25 minutes)
1. Open `core/pipeline_dataset.py`
2. Find the `ImageItem` class (line ~15)
3. **Task**: Add a new field called `processing_notes: str = ""`
4. **Challenge**: Where else would you need to update code?

## Exercise Set 2: Testing and Debugging

### Ex 2.1: Running Tests (15 minutes)
```bash
# Run specific test categories
python tests/run_all_tests.py --category dataset
python tests/run_all_tests.py --category adapters --verbose
```
**Question**: Which tests pass/fail and what do the failures teach you?

### Ex 2.2: Mock Understanding (30 minutes)
1. Open `tests/test_adapters.py:84-103`
2. Study the mock setup for mask generation
3. **Task**: Create a similar mock for a new feature type
4. **Question**: Why use mocks instead of real components?

### Ex 2.3: Error Simulation (20 minutes)
1. Temporarily rename a required file in the test dataset
2. Run tests and observe error handling
3. **Question**: How does the pipeline handle missing files?

## Exercise Set 3: Extending the Pipeline

### Ex 3.1: Add New Visual Feature (45 minutes)
1. Open `config/pipeline_config.yaml`
2. Add `"building"` to `target_features`
3. **Predict**: What will break when you run tests?
4. **Fix**: Update the mock data in tests to include building labels

### Ex 3.2: Custom Selection Algorithm (30 minutes)
1. Open `core/pipeline_dataset.py:200-250`
2. Find the selection methods
3. **Task**: Add a `random` selection method
4. **Test**: Add test case for your new method

### Ex 3.3: Enhanced Logging (25 minutes)
1. Open any adapter file
2. Find existing logging statements
3. **Task**: Add debug logging for processing times
4. **Question**: When would detailed logging be crucial in research?

## Exercise Set 4: Research Methodology

### Ex 4.1: Experimental Design (20 minutes)
**Study the systematic selection algorithms:**
1. `first_n`: Takes first N images
2. `systematic`: Takes every Kth image  
3. `evenly_spaced`: Distributes across available range
**Question**: Which method would give most representative samples?

### Ex 4.2: Reproducibility Analysis (25 minutes)
1. Run the same configuration twice
2. Compare results
3. **Question**: What ensures identical results across runs?
4. **Challenge**: What could cause non-reproducible results?

### Ex 4.3: Ground Truth Validation (30 minutes)
1. Open `adapters/evaluator_adapter.py:400-450`
2. Study the distance calculation method
3. **Task**: Add validation for coordinate bounds
4. **Question**: How would you handle GPS coordinate errors?

## Exercise Set 5: Performance and Scalability

### Ex 5.1: Resource Monitoring (15 minutes)
1. Run: `python -c "from core.pipeline_utils import monitor_system_resources; print(monitor_system_resources())"`
2. **Observe**: CPU and memory usage patterns
3. **Question**: When would you need to limit resource usage?

### Ex 5.2: Batch Size Optimization (30 minutes)
1. Open configuration and change `batch_size` from 2 to 1
2. **Predict**: How will this affect processing time?
3. **Question**: What's the trade-off between batch size and memory?

### Ex 5.3: Async Processing Understanding (35 minutes)
1. Open `adapters/evaluator_adapter.py:150-200`
2. Study the async evaluation methods
3. **Question**: Why use async for API calls but not for image processing?
4. **Challenge**: Where else could async improve performance?

## Research Project Ideas

### Beginner Projects:
1. **Dataset Analysis**: Create visualizations of GPS coordinate distributions
2. **Error Analysis**: Study which types of images cause processing failures
3. **Performance Benchmarking**: Compare processing times across different configurations

### Intermediate Projects:
1. **Feature Comparison**: Analyze which visual features most impact geolocation accuracy
2. **Selection Method Study**: Compare systematic vs. random image selection effectiveness
3. **Scalability Testing**: Test pipeline performance with larger datasets

### Advanced Projects:
1. **New Visual Features**: Integrate additional object detection for new feature types
2. **Alternative Evaluation**: Replace GPT-4 Vision with other geolocation models
3. **Distributed Processing**: Implement multi-machine processing capabilities

## Learning Checkpoints

After each exercise set, ask yourself:
- **Understanding**: Can I explain this component to someone else?
- **Application**: Could I modify this for a different research question?
- **Integration**: How does this connect to the broader pipeline?
- **Research Value**: What research insights does this component enable?

## Common Pitfalls for Beginners

1. **Don't Skip Configuration**: Always understand what each config parameter does
2. **Read Error Messages**: Pipeline provides detailed error information
3. **Use Tests**: Run tests frequently to catch issues early
4. **Check File Paths**: Many issues stem from incorrect file path assumptions
5. **Resource Limits**: Monitor memory/GPU usage during development

## Getting Help

1. **Documentation**: Start with README.md and inline docstrings
2. **Tests**: Test files show expected behavior and usage patterns
3. **Configuration**: Default configs show working parameter combinations
4. **Logging**: Enable debug logging to see detailed execution flow
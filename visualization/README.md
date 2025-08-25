# Academic Visualization Generator

This toolkit generates publication-ready visualizations and statistical analysis for geolocation evaluation results from the Visual Cue Evaluation Pipeline.

## Features

- **Adaptive Configuration**: Automatically reads pipeline configuration to adapt to different experimental setups
- **Visual Comparisons**: Creates side-by-side image grids showing original vs feature-removed images
- **Statistical Analysis**: Generates comprehensive charts and statistics for error analysis
- **Publication Ready**: All outputs are formatted for academic presentations (300 DPI, proper styling)
- **Flexible Usage**: Can generate all visualizations or focus on specific components

## Quick Start

### Basic Usage
```bash
# Generate all visualizations and analysis
python visualization/main.py --summary

# Generate only visual comparisons
python visualization/main.py --visual-only

# Generate only statistical charts
python visualization/main.py --charts-only
```

### Requirements
```bash
pip install -r visualization/requirements.txt
```

## Output Structure

```
visualization/output/
├── image_comparisons/           # Visual comparison grids
│   ├── [image_id]_comparison.png  # Individual comparisons
│   └── summary_comparison_grid.png # Overview of all images
├── error_analysis/              # Statistical analysis
│   ├── error_comparison_chart.png      # Side-by-side error bars
│   ├── error_increase_analysis.png     # Impact analysis
│   ├── city_comparison_chart.png       # Baseline by city
│   └── statistics_report.txt          # Comprehensive stats
└── generation_summary.txt       # Summary of generated files
```

## Configuration Adaptation

The system automatically adapts to your pipeline configuration:

- **Cities**: Works with any city list (`dataset.cities`)
- **Images per City**: Handles any number (`dataset.images_per_city`) 
- **Target Features**: Adapts to any feature set (`features.target_features`)
- **Output Paths**: Follows your output directory structure

## Generated Visualizations

### 1. Visual Comparisons
- **Individual Grids**: Each image with original + all feature removals
- **Summary Grid**: Overview showing multiple images side-by-side
- Automatically handles missing images gracefully

### 2. Statistical Charts
- **Error Comparison**: Mean error by feature with confidence intervals
- **Error Increase Analysis**: Shows impact of each feature removal
- **Impact Distribution**: Percentage of cases improved/degraded
- **City Comparison**: Baseline performance across locations

### 3. Statistical Reports
- Comprehensive statistics for each feature removal
- Mean, median, standard deviation, ranges
- Improvement/degradation rates
- City-level baseline performance metrics

## Academic Usage

### For Presentations
- Use `summary_comparison_grid.png` for overview slides
- Include individual comparison images for detailed case studies
- Reference statistical charts for quantitative evidence

### For Papers
- All images are 300 DPI PNG format (publication ready)
- Statistics report provides numerical values for citations
- Charts use academic styling (serif fonts, proper labels)

### Key Metrics
- **error_km**: Absolute distance error in kilometers
- **error_increase_km**: Change from baseline (negative = improvement)
- **improvement_rate**: Percentage of cases where removal helped
- **degradation_rate**: Percentage of cases where removal hurt

## Customization

### Color Schemes
Edit the color palettes in `create_error_charts.py`:
```python
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
```

### Figure Sizes
Adjust figure dimensions in the plotting functions:
```python
fig, ax = plt.subplots(figsize=(12, 8))  # width, height in inches
```

### Font Settings
Modify academic styling in `ErrorAnalysisGenerator.__init__()`:
```python
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    # ...
})
```

## Troubleshooting

### Missing Images
- Visual comparisons gracefully handle missing CLIPAway results
- Shows "not found" placeholders for missing files
- Check pipeline output directories for completeness

### No Data
- Ensure evaluation results exist in `pipeline/output/evaluation_results/`
- Verify JSON format matches expected structure
- Check that pipeline completed successfully

### Import Errors
- Install requirements: `pip install -r requirements.txt`
- Some optional dependencies (like seaborn) are commented out if not available

## Example Output

The system generates visualizations for experimental results showing:

- **Baseline Performance**: GPT-4 geolocation accuracy without modifications
- **Feature Impact**: How removing humans, vehicles, objects, or construction affects accuracy
- **City Variations**: Different baseline performance across German cities
- **Statistical Significance**: Error bars and confidence measures

Perfect for academic presentations demonstrating the impact of visual features on AI geolocation capabilities.
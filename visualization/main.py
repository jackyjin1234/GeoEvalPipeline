#!/usr/bin/env python3
"""
Academic Visualization Generator for Geolocation Evaluation Pipeline

Main entry point for generating comprehensive visualizations and analysis
for presenting geolocation evaluation results to academic audiences.

Usage:
    python visualization/main.py [options]
    
Options:
    --config PATH     Path to pipeline config (default: auto-detect)
    --visual-only     Generate only visual comparisons
    --charts-only     Generate only error charts
    --summary         Create summary report
    --help           Show this help message
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config_reader import ConfigReader
from create_visual_comparisons import VisualComparisonGenerator
from create_error_charts import ErrorAnalysisGenerator


def print_banner():
    """Print application banner."""
    print("=" * 80)
    print("GEOLOCATION EVALUATION - ACADEMIC VISUALIZATION GENERATOR")
    print("=" * 80)
    print("Generates publication-ready visualizations and statistical analysis")
    print("for geolocation accuracy evaluation with visual feature removal.")
    print("=" * 80)


def create_summary_report(config: ConfigReader, visual_files: list, chart_files: list):
    """Create a summary report of all generated files."""
    summary_path = Path("/scratch/kesuniot_root/kesuniot0/jackyjin/GeoEvalPipeline/visualization/output/generation_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("VISUALIZATION GENERATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Cities: {', '.join(config.cities)} ({len(config.cities)} total)\n")
        f.write(f"Images per city: {config.images_per_city}\n")
        f.write(f"Total expected images: {config.get_expected_image_count()}\n")
        f.write(f"Target features: {', '.join(config.target_features)} ({len(config.target_features)} total)\n")
        f.write(f"Output directory: {config.output_base_directory}\n\n")
        
        f.write("GENERATED FILES:\n")
        f.write("-" * 20 + "\n")
        
        if visual_files:
            f.write(f"Visual Comparisons ({len(visual_files)} files):\n")
            for file_path in visual_files:
                f.write(f"  - {file_path.name}\n")
            f.write("\n")
        
        if chart_files:
            f.write(f"Statistical Charts and Reports ({len(chart_files)} files):\n")
            for file_path in chart_files:
                f.write(f"  - {file_path.name}\n")
            f.write("\n")
        
        total_files = len(visual_files) + len(chart_files)
        f.write(f"TOTAL GENERATED FILES: {total_files}\n\n")
        
        f.write("OUTPUT STRUCTURE:\n")
        f.write("-" * 20 + "\n")
        f.write("visualization/output/\n")
        f.write("├── image_comparisons/    # Side-by-side visual comparisons\n")
        f.write("├── error_analysis/       # Statistical charts and reports\n")
        f.write("└── generation_summary.txt # This summary file\n\n")
        
        f.write("USAGE RECOMMENDATIONS:\n")
        f.write("-" * 25 + "\n")
        f.write("• Use individual comparison images for detailed case studies\n")
        f.write("• Include summary comparison grid in presentations\n")
        f.write("• Reference error analysis charts for statistical evidence\n")
        f.write("• Cite statistics from the statistics report\n")
        f.write("• All images are publication-ready (300 DPI PNG format)\n")
    
    return summary_path


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Generate academic visualizations for geolocation evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualization/main.py                    # Generate all visualizations
  python visualization/main.py --visual-only     # Only create image comparisons  
  python visualization/main.py --charts-only     # Only create statistical charts
  python visualization/main.py --summary         # Include detailed summary report
        """
    )
    
    parser.add_argument('--config', type=str, 
                       help='Path to pipeline config file (default: auto-detect)')
    parser.add_argument('--visual-only', action='store_true',
                       help='Generate only visual comparisons')
    parser.add_argument('--charts-only', action='store_true', 
                       help='Generate only error charts and statistics')
    parser.add_argument('--summary', action='store_true',
                       help='Create detailed summary report')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Load configuration
    try:
        config = ConfigReader(args.config) if args.config else ConfigReader()
        config.print_config_summary()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    visual_files = []
    chart_files = []
    
    # Generate visual comparisons
    if not args.charts_only:
        print("\n" + "=" * 60)
        print("GENERATING VISUAL COMPARISONS")
        print("=" * 60)
        
        try:
            visual_generator = VisualComparisonGenerator(config)
            
            # Create individual comparison grids
            individual_files = visual_generator.create_all_comparisons()
            visual_files.extend(individual_files)
            
            # Create summary grid
            summary_file = visual_generator.create_summary_grid()
            if summary_file:
                visual_files.append(summary_file)
            
            print(f"\n✓ Visual comparisons complete: {len(visual_files)} files generated")
            
        except Exception as e:
            print(f"✗ Error generating visual comparisons: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate error analysis charts
    if not args.visual_only:
        print("\n" + "=" * 60)
        print("GENERATING ERROR ANALYSIS")
        print("=" * 60)
        
        try:
            chart_generator = ErrorAnalysisGenerator(config)
            chart_files = chart_generator.create_all_charts()
            
            print(f"\n✓ Error analysis complete: {len(chart_files)} files generated")
            
        except Exception as e:
            print(f"✗ Error generating charts and analysis: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary report if requested
    if args.summary:
        print("\n" + "=" * 60)
        print("GENERATING SUMMARY REPORT")
        print("=" * 60)
        
        try:
            summary_path = create_summary_report(config, visual_files, chart_files)
            print(f"✓ Summary report created: {summary_path}")
        except Exception as e:
            print(f"✗ Error creating summary report: {e}")
    
    # Final summary
    total_files = len(visual_files) + len(chart_files)
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total files generated: {total_files}")
    
    if visual_files:
        print(f"Visual comparisons: {len(visual_files)} files")
        print(f"  Location: visualization/output/image_comparisons/")
    
    if chart_files:
        print(f"Statistical analysis: {len(chart_files)} files")  
        print(f"  Location: visualization/output/error_analysis/")
    
    if total_files > 0:
        print(f"\nAll files are ready for academic presentation!")
        print(f"Output directory: visualization/output/")
    else:
        print(f"\nNo files were generated. Check your pipeline output data.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
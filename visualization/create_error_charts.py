"""
Error analysis and chart generator for geolocation evaluation pipeline.
Creates statistical charts and analysis of error_km and error_increase_km metrics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
# import seaborn as sns  # Optional, not used in current implementation

from config_reader import ConfigReader


class ErrorAnalysisGenerator:
    """Generates error analysis charts and statistics."""
    
    def __init__(self, config_reader: ConfigReader):
        """
        Initialize the error analysis generator.
        
        Args:
            config_reader: ConfigReader instance with pipeline configuration
        """
        self.config = config_reader
        self.output_paths = config_reader.get_output_paths()
        self.output_dir = Path("/scratch/kesuniot_root/kesuniot0/jackyjin/GeoEvalPipeline/visualization/output/error_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style for academic presentation
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 18
        })
    
    def load_evaluation_results(self) -> Dict:
        """Load evaluation results from JSON file."""
        try:
            with open(self.output_paths['evaluation_results'], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error: Could not load evaluation results: {e}")
            return {}
    
    def extract_error_data(self) -> pd.DataFrame:
        """Extract error data into a structured DataFrame."""
        results = self.load_evaluation_results()
        
        if not results:
            print("No evaluation results found.")
            return pd.DataFrame()
        
        data_rows = []
        
        for image_id, image_data in results.items():
            # Extract city from image_id
            city = image_id.split('_')[0]
            ground_truth = image_data.get('ground_truth_coords', [None, None])
            
            for feature_result in image_data.get('feature_results', []):
                feature = feature_result.get('feature_removed', 'unknown')
                error_km = feature_result.get('error_km', 0)
                error_increase_km = feature_result.get('error_increase_km', 0)
                
                gpt_response = feature_result.get('gpt_response', {})
                confidence = gpt_response.get('confidence', 'Unknown')
                predicted_coords = [gpt_response.get('latitude'), gpt_response.get('longitude')]
                
                data_rows.append({
                    'image_id': image_id,
                    'city': city,
                    'feature_removed': feature,
                    'error_km': error_km,
                    'error_increase_km': error_increase_km,
                    'confidence': confidence,
                    'ground_truth_lat': ground_truth[0],
                    'ground_truth_lon': ground_truth[1],
                    'predicted_lat': predicted_coords[0],
                    'predicted_lon': predicted_coords[1]
                })
        
        return pd.DataFrame(data_rows)
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics for error analysis."""
        if df.empty:
            return {}
        
        stats = {}
        
        # Overall statistics
        baseline_df = df[df['feature_removed'] == 'none']
        feature_df = df[df['feature_removed'] != 'none']
        
        stats['overall'] = {
            'total_images': df['image_id'].nunique(),
            'total_evaluations': len(df),
            'baseline_mean_error': baseline_df['error_km'].mean(),
            'baseline_median_error': baseline_df['error_km'].median(),
            'baseline_std_error': baseline_df['error_km'].std(),
        }
        
        # Statistics by feature
        stats['by_feature'] = {}
        features = [f for f in df['feature_removed'].unique() if f != 'none']
        
        for feature in features:
            feature_data = df[df['feature_removed'] == feature]
            
            stats['by_feature'][feature] = {
                'mean_error': feature_data['error_km'].mean(),
                'median_error': feature_data['error_km'].median(),
                'std_error': feature_data['error_km'].std(),
                'min_error': feature_data['error_km'].min(),
                'max_error': feature_data['error_km'].max(),
                'mean_error_increase': feature_data['error_increase_km'].mean(),
                'median_error_increase': feature_data['error_increase_km'].median(),
                'std_error_increase': feature_data['error_increase_km'].std(),
                'min_error_increase': feature_data['error_increase_km'].min(),
                'max_error_increase': feature_data['error_increase_km'].max(),
                'cases_improved': (feature_data['error_increase_km'] < 0).sum(),
                'cases_degraded': (feature_data['error_increase_km'] > 0).sum(),
                'cases_unchanged': (feature_data['error_increase_km'] == 0).sum(),
                'improvement_rate': (feature_data['error_increase_km'] < 0).mean() * 100,
                'degradation_rate': (feature_data['error_increase_km'] > 0).mean() * 100
            }
        
        # Statistics by city
        stats['by_city'] = {}
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            baseline_city = city_data[city_data['feature_removed'] == 'none']
            
            stats['by_city'][city] = {
                'mean_baseline_error': baseline_city['error_km'].mean(),
                'images_count': city_data['image_id'].nunique(),
                'total_evaluations': len(city_data)
            }
        
        return stats
    
    def create_error_comparison_chart(self, df: pd.DataFrame) -> Path:
        """Create side-by-side bar chart comparing error_km across features."""
        if df.empty:
            return None
        
        # Prepare data for plotting
        features = ['none'] + [f for f in self.config.target_features if f in df['feature_removed'].values]
        feature_labels = ['Baseline'] + [f.title() for f in features[1:]]
        
        # Calculate means and standard errors
        means = []
        errors = []
        
        for feature in features:
            feature_data = df[df['feature_removed'] == feature]['error_km']
            means.append(feature_data.mean())
            errors.append(feature_data.std() / np.sqrt(len(feature_data)) if len(feature_data) > 1 else 0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x_pos = np.arange(len(features))
        colors = ['#2E86AB'] + ['#A23B72', '#F18F01', '#C73E1D', '#592E83'][:len(features)-1]
        
        bars = ax.bar(x_pos, means, yerr=errors, capsize=5, color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the plot
        ax.set_xlabel('Feature Condition', fontweight='bold')
        ax.set_ylabel('Mean Geolocation Error (km)', fontweight='bold')
        ax.set_title('Geolocation Error by Feature Removal Condition', fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean_val, err_val in zip(bars, means, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err_val + max(means)*0.01,
                   f'{mean_val:.1f}±{err_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "error_comparison_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def create_error_increase_chart(self, df: pd.DataFrame) -> Path:
        """Create chart showing error increase/decrease for each feature."""
        if df.empty:
            return None
        
        # Filter out baseline (none) condition
        feature_df = df[df['feature_removed'] != 'none'].copy()
        
        if feature_df.empty:
            return None
        
        # Create grouped bar chart
        features = [f for f in self.config.target_features if f in feature_df['feature_removed'].values]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Chart 1: Mean error increase with error bars
        means = []
        stds = []
        feature_labels = []
        
        for feature in features:
            feature_data = feature_df[feature_df['feature_removed'] == feature]['error_increase_km']
            means.append(feature_data.mean())
            stds.append(feature_data.std() / np.sqrt(len(feature_data)) if len(feature_data) > 1 else 0)
            feature_labels.append(feature.title())
        
        x_pos = np.arange(len(features))
        colors = ['#A23B72', '#F18F01', '#C73E1D', '#592E83'][:len(features)]
        
        bars1 = ax1.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Feature Removed', fontweight='bold')
        ax1.set_ylabel('Mean Error Increase (km)', fontweight='bold')
        ax1.set_title('Error Increase by Feature Removal', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(feature_labels, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars1, means, stds):
            height = bar.get_height()
            y_pos = height + std_val + abs(max(means) - min(means))*0.02 if height >= 0 else height - std_val - abs(max(means) - min(means))*0.02
            ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{mean_val:.1f}±{std_val:.1f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # Chart 2: Improvement/Degradation rates
        improvement_rates = []
        degradation_rates = []
        
        for feature in features:
            feature_data = feature_df[feature_df['feature_removed'] == feature]['error_increase_km']
            total_cases = len(feature_data)
            improved = (feature_data < 0).sum()
            degraded = (feature_data > 0).sum()
            
            improvement_rates.append(improved / total_cases * 100)
            degradation_rates.append(degraded / total_cases * 100)
        
        width = 0.35
        bars2_1 = ax2.bar([x - width/2 for x in x_pos], improvement_rates, width, 
                         label='Improved Accuracy', color='#2E8B57', alpha=0.8)
        bars2_2 = ax2.bar([x + width/2 for x in x_pos], degradation_rates, width,
                         label='Degraded Accuracy', color='#DC143C', alpha=0.8)
        
        ax2.set_xlabel('Feature Removed', fontweight='bold')
        ax2.set_ylabel('Percentage of Cases (%)', fontweight='bold')
        ax2.set_title('Impact Distribution by Feature Removal', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(feature_labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        # Add percentage labels
        for bars, values in [(bars2_1, improvement_rates), (bars2_2, degradation_rates)]:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "error_increase_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def create_city_comparison_chart(self, df: pd.DataFrame) -> Path:
        """Create chart comparing baseline performance across cities."""
        if df.empty:
            return None
        
        baseline_df = df[df['feature_removed'] == 'none'].copy()
        
        if baseline_df.empty:
            return None
        
        # Group by city
        city_stats = baseline_df.groupby('city')['error_km'].agg(['mean', 'std', 'count']).reset_index()
        city_stats['stderr'] = city_stats['std'] / np.sqrt(city_stats['count'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(city_stats))
        colors = plt.cm.Set3(np.linspace(0, 1, len(city_stats)))
        
        bars = ax.bar(x_pos, city_stats['mean'], yerr=city_stats['stderr'], 
                     capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('City', fontweight='bold')
        ax.set_ylabel('Mean Baseline Error (km)', fontweight='bold')
        ax.set_title('Baseline Geolocation Performance by City', fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([city.title() for city in city_stats['city']])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels
        for bar, mean_val, stderr_val in zip(bars, city_stats['mean'], city_stats['stderr']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stderr_val + city_stats['mean'].max()*0.02,
                   f'{mean_val:.1f}±{stderr_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "city_comparison_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def generate_statistics_report(self, stats: Dict) -> Path:
        """Generate a comprehensive statistics report."""
        output_path = self.output_dir / "statistics_report.txt"
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GEOLOCATION EVALUATION STATISTICAL REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            if 'overall' in stats:
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 30 + "\n")
                overall = stats['overall']
                f.write(f"Total Images: {overall.get('total_images', 'N/A')}\n")
                f.write(f"Total Evaluations: {overall.get('total_evaluations', 'N/A')}\n")
                f.write(f"Baseline Mean Error: {overall.get('baseline_mean_error', 0):.2f} km\n")
                f.write(f"Baseline Median Error: {overall.get('baseline_median_error', 0):.2f} km\n")
                f.write(f"Baseline Std Dev: {overall.get('baseline_std_error', 0):.2f} km\n\n")
            
            # Feature statistics
            if 'by_feature' in stats:
                f.write("STATISTICS BY FEATURE REMOVAL\n")
                f.write("-" * 40 + "\n")
                
                for feature, feature_stats in stats['by_feature'].items():
                    f.write(f"\n{feature.upper()} REMOVAL:\n")
                    f.write(f"  Mean Error: {feature_stats['mean_error']:.2f} ± {feature_stats['std_error']:.2f} km\n")
                    f.write(f"  Median Error: {feature_stats['median_error']:.2f} km\n")
                    f.write(f"  Error Range: {feature_stats['min_error']:.2f} - {feature_stats['max_error']:.2f} km\n")
                    f.write(f"  Mean Error Increase: {feature_stats['mean_error_increase']:.2f} ± {feature_stats['std_error_increase']:.2f} km\n")
                    f.write(f"  Median Error Increase: {feature_stats['median_error_increase']:.2f} km\n")
                    f.write(f"  Cases Improved: {feature_stats['cases_improved']} ({feature_stats['improvement_rate']:.1f}%)\n")
                    f.write(f"  Cases Degraded: {feature_stats['cases_degraded']} ({feature_stats['degradation_rate']:.1f}%)\n")
                    f.write(f"  Cases Unchanged: {feature_stats['cases_unchanged']}\n")
                
                f.write("\n")
            
            # City statistics
            if 'by_city' in stats:
                f.write("STATISTICS BY CITY\n")
                f.write("-" * 25 + "\n")
                
                for city, city_stats in stats['by_city'].items():
                    f.write(f"\n{city.upper()}:\n")
                    f.write(f"  Mean Baseline Error: {city_stats['mean_baseline_error']:.2f} km\n")
                    f.write(f"  Images Processed: {city_stats['images_count']}\n")
                    f.write(f"  Total Evaluations: {city_stats['total_evaluations']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        return output_path
    
    def create_all_charts(self) -> List[Path]:
        """Create all error analysis charts and reports."""
        print("=== Error Analysis Generator ===")
        
        # Load and process data
        df = self.extract_error_data()
        
        if df.empty:
            print("No data available for analysis.")
            return []
        
        print(f"Loaded data: {len(df)} evaluations from {df['image_id'].nunique()} images")
        
        # Calculate statistics
        stats = self.calculate_statistics(df)
        
        created_files = []
        
        # Generate charts
        print("Generating error comparison chart...")
        chart_path = self.create_error_comparison_chart(df)
        if chart_path:
            created_files.append(chart_path)
            print(f"  ✓ Created: {chart_path.name}")
        
        print("Generating error increase analysis...")
        increase_path = self.create_error_increase_chart(df)
        if increase_path:
            created_files.append(increase_path)
            print(f"  ✓ Created: {increase_path.name}")
        
        print("Generating city comparison chart...")
        city_path = self.create_city_comparison_chart(df)
        if city_path:
            created_files.append(city_path)
            print(f"  ✓ Created: {city_path.name}")
        
        # Generate statistics report
        print("Generating statistics report...")
        report_path = self.generate_statistics_report(stats)
        created_files.append(report_path)
        print(f"  ✓ Created: {report_path.name}")
        
        print(f"\n=== Analysis Complete ===")
        print(f"Generated {len(created_files)} files in {self.output_dir}")
        
        return created_files


def main():
    """Main function to generate error analysis."""
    # Load configuration
    config = ConfigReader()
    config.print_config_summary()
    
    # Create analyzer
    analyzer = ErrorAnalysisGenerator(config)
    
    # Generate all charts and reports
    created_files = analyzer.create_all_charts()
    
    print(f"\nOutput files:")
    for file_path in created_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
"""
Visual comparison generator for geolocation evaluation pipeline.
Creates side-by-side image grids showing original images and feature-removed versions.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

from config_reader import ConfigReader


class VisualComparisonGenerator:
    """Generates visual comparisons of original and feature-removed images."""
    
    def __init__(self, config_reader: ConfigReader):
        """
        Initialize the visual comparison generator.
        
        Args:
            config_reader: ConfigReader instance with pipeline configuration
        """
        self.config = config_reader
        self.output_paths = config_reader.get_output_paths()
        self.output_dir = Path("/scratch/kesuniot_root/kesuniot0/jackyjin/GeoEvalPipeline/visualization/output/image_comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_evaluation_results(self) -> Dict:
        """Load evaluation results to get list of processed images."""
        try:
            with open(self.output_paths['evaluation_results'], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load evaluation results: {e}")
            return {}
    
    def find_original_image(self, image_id: str) -> Optional[Path]:
        """Find the original processed image (merged left and right halves)."""
        # Try to find left and right processed images to merge them
        original_dir = self.output_paths['processed_original']
        left_path = original_dir / f"{image_id}_left.png"
        right_path = original_dir / f"{image_id}_right.png"
        
        if left_path.exists() and right_path.exists():
            return self._merge_image_halves(left_path, right_path, image_id)
        
        # Fallback: try to find a merged original if it exists
        merged_path = original_dir / f"{image_id}_merged.png"
        if merged_path.exists():
            return merged_path
            
        return None
    
    def _merge_image_halves(self, left_path: Path, right_path: Path, image_id: str) -> Path:
        """Merge left and right image halves into a single image."""
        try:
            left_img = Image.open(left_path)
            right_img = Image.open(right_path)
            
            # Create merged image
            total_width = left_img.width + right_img.width
            merged_img = Image.new('RGB', (total_width, left_img.height))
            merged_img.paste(left_img, (0, 0))
            merged_img.paste(right_img, (left_img.width, 0))
            
            # Save merged image for reuse
            merged_path = self.output_dir / f"{image_id}_original_merged.png"
            merged_img.save(merged_path)
            
            return merged_path
        except Exception as e:
            print(f"Warning: Could not merge image halves for {image_id}: {e}")
            return left_path  # Fallback to left half only
    
    def find_feature_removed_images(self, image_id: str) -> Dict[str, Optional[Path]]:
        """Find all feature-removed versions of an image."""
        clipaway_dir = self.output_paths['clipaway_results']
        feature_images = {}
        
        for feature in self.config.target_features:
            feature_path = clipaway_dir / feature / f"{image_id}_{feature}_removed.jpg"
            if feature_path.exists():
                feature_images[feature] = feature_path
            else:
                feature_images[feature] = None
                
        return feature_images
    
    def create_comparison_grid(self, image_id: str, save_individual: bool = True) -> Optional[Path]:
        """
        Create a comparison grid for a single image showing original and all feature removals.
        
        Args:
            image_id: ID of the image to create comparison for
            save_individual: Whether to save individual comparison grids
            
        Returns:
            Path to the saved comparison grid, or None if failed
        """
        # Find original and feature-removed images
        original_path = self.find_original_image(image_id)
        feature_images = self.find_feature_removed_images(image_id)
        
        if original_path is None:
            print(f"Warning: Could not find original image for {image_id}")
            return None
        
        # Count available feature images
        available_features = [f for f, path in feature_images.items() if path is not None]
        if not available_features:
            print(f"Warning: No feature-removed images found for {image_id}")
            return None
        
        # Create figure with appropriate layout
        n_features = len(available_features)
        n_cols = min(5, n_features + 1)  # +1 for original, max 5 columns
        n_rows = max(1, (n_features + 1) // n_cols + ((n_features + 1) % n_cols > 0))
        
        fig_width = n_cols * 4
        fig_height = n_rows * 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Load and display original image
        try:
            original_img = Image.open(original_path)
            axes[0, 0].imshow(original_img)
            axes[0, 0].set_title('Original', fontsize=12, weight='bold')
            axes[0, 0].axis('off')
        except Exception as e:
            print(f"Warning: Could not load original image {original_path}: {e}")
            axes[0, 0].text(0.5, 0.5, 'Original\n(not found)', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].axis('off')
        
        # Load and display feature-removed images
        idx = 1
        for feature in available_features:
            row = idx // n_cols
            col = idx % n_cols
            
            try:
                feature_img = Image.open(feature_images[feature])
                axes[row, col].imshow(feature_img)
                axes[row, col].set_title(f'{feature.title()}\nRemoved', fontsize=12, weight='bold')
                axes[row, col].axis('off')
            except Exception as e:
                print(f"Warning: Could not load feature image {feature_images[feature]}: {e}")
                axes[row, col].text(0.5, 0.5, f'{feature.title()}\nRemoved\n(not found)', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
            
            idx += 1
        
        # Hide unused subplots
        total_plots = n_features + 1
        for i in range(total_plots, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Visual Feature Removal Comparison: {image_id}', fontsize=16, weight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if save_individual:
            output_path = self.output_dir / f"{image_id}_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return output_path
        
        return None
    
    def create_all_comparisons(self) -> List[Path]:
        """Create comparison grids for all available images."""
        evaluation_results = self.load_evaluation_results()
        
        if not evaluation_results:
            print("No evaluation results found. Cannot create comparisons.")
            return []
        
        image_ids = list(evaluation_results.keys())
        print(f"Creating visual comparisons for {len(image_ids)} images...")
        
        created_files = []
        for i, image_id in enumerate(image_ids, 1):
            print(f"Processing {i}/{len(image_ids)}: {image_id}")
            
            comparison_path = self.create_comparison_grid(image_id)
            if comparison_path:
                created_files.append(comparison_path)
                print(f"  ✓ Created: {comparison_path.name}")
            else:
                print(f"  ✗ Failed to create comparison for {image_id}")
        
        print(f"\nCompleted: {len(created_files)}/{len(image_ids)} comparison grids created")
        return created_files
    
    def create_summary_grid(self, max_images: int = 6) -> Optional[Path]:
        """Create a summary grid showing comparisons for a subset of images."""
        evaluation_results = self.load_evaluation_results()
        
        if not evaluation_results:
            print("No evaluation results found. Cannot create summary grid.")
            return None
        
        image_ids = list(evaluation_results.keys())[:max_images]
        n_images = len(image_ids)
        n_features = len(self.config.target_features)
        
        # Create large grid: rows for images, columns for original + features
        fig_width = (n_features + 1) * 3
        fig_height = n_images * 2.5
        
        fig, axes = plt.subplots(n_images, n_features + 1, figsize=(fig_width, fig_height))
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for row, image_id in enumerate(image_ids):
            # Original image
            original_path = self.find_original_image(image_id)
            if original_path and original_path.exists():
                try:
                    original_img = Image.open(original_path)
                    axes[row, 0].imshow(original_img)
                except:
                    axes[row, 0].text(0.5, 0.5, 'Original\n(not found)', ha='center', va='center', 
                                    transform=axes[row, 0].transAxes)
            else:
                axes[row, 0].text(0.5, 0.5, 'Original\n(not found)', ha='center', va='center', 
                                transform=axes[row, 0].transAxes)
            
            axes[row, 0].set_title('Original' if row == 0 else '', fontsize=10, weight='bold')
            axes[row, 0].set_ylabel(image_id, rotation=0, ha='right', va='center', fontsize=8)
            axes[row, 0].axis('off')
            
            # Feature-removed images
            feature_images = self.find_feature_removed_images(image_id)
            for col, feature in enumerate(self.config.target_features, 1):
                if feature in feature_images and feature_images[feature]:
                    try:
                        feature_img = Image.open(feature_images[feature])
                        axes[row, col].imshow(feature_img)
                    except:
                        axes[row, col].text(0.5, 0.5, f'{feature.title()}\nRemoved\n(error)', 
                                          ha='center', va='center', transform=axes[row, col].transAxes, fontsize=8)
                else:
                    axes[row, col].text(0.5, 0.5, f'{feature.title()}\nRemoved\n(not found)', 
                                      ha='center', va='center', transform=axes[row, col].transAxes, fontsize=8)
                
                axes[row, col].set_title(f'{feature.title()}\nRemoved' if row == 0 else '', fontsize=10, weight='bold')
                axes[row, col].axis('off')
        
        plt.suptitle('Visual Feature Removal Comparison Summary', fontsize=16, weight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, left=0.1)
        
        output_path = self.output_dir / "summary_comparison_grid.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Created summary comparison grid: {output_path}")
        return output_path


def main():
    """Main function to generate visual comparisons."""
    print("=== Visual Comparison Generator ===")
    
    # Load configuration
    config = ConfigReader()
    config.print_config_summary()
    
    # Create generator
    generator = VisualComparisonGenerator(config)
    
    # Generate individual comparison grids
    created_files = generator.create_all_comparisons()
    
    # Generate summary grid
    summary_path = generator.create_summary_grid()
    
    print(f"\n=== Generation Complete ===")
    print(f"Individual comparisons: {len(created_files)} files")
    if summary_path:
        print(f"Summary grid: {summary_path}")
    print(f"Output directory: {generator.output_dir}")


if __name__ == "__main__":
    main()
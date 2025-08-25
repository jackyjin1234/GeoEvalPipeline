#!/usr/bin/env python3
"""
Image Processor Adapter

This adapter integrates the existing image_processor.py functionality
with the optimized pipeline architecture, providing batch processing
for image splitting and scaling operations.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from utils.image_processor import downscale, upscale

from core.pipeline_dataset import ImageItem

logger = logging.getLogger(__name__)


class ImageProcessorAdapter:
    """
    Adapter for image processing functionality optimized for pipeline processing.
    
    Handles batch processing of image splitting, downscaling, and upscaling
    for both original images and generated masks.
    """
    
    def __init__(self, target_size: int = 512):
        """
        Initialize image processor adapter.
        
        Args:
            target_size: Target size for processed images (e.g., 512 for 512x512)
            downscale_interpolation: Interpolation method for downscaling
            upscale_interpolation: Interpolation method for upscaling
        """
        self.target_size = target_size
        # self.downscale_interpolation = downscale_interpolation
        # self.upscale_interpolation = upscale_interpolation
        
        # Validate interpolation methods
        # valid_methods = ["area", "cubic", "linear", "nearest"]
        # if downscale_interpolation not in valid_methods:
        #     raise ValueError(f"Invalid downscale interpolation: {downscale_interpolation}")
        # if upscale_interpolation not in valid_methods:
        #     raise ValueError(f"Invalid upscale interpolation: {upscale_interpolation}")
        
        logger.info(f"ImageProcessorAdapter initialized - Target size: {target_size}x{target_size}")
    
    def process_images_batch(self, items: List[ImageItem], output_dir: Path,
                           process_masks: bool = True) -> Dict[str, List[Path]]:
        """
        Process a batch of images and their masks.
        
        Args:
            items: List of ImageItem objects to process
            output_dir: Base output directory for processed images
            process_masks: Whether to also process associated masks
        
        Returns:
            Dictionary with processing results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'original_left': [],
            'original_right': [],
            'mask_results': {}
        }
        
        failed_items = []
        
        logger.info(f"Starting batch image processing for {len(items)} items")
        
        for i, item in enumerate(items):
            try:
                # Process original image
                left_path, right_path = self._process_original_image(item, output_dir)
                results['original_left'].append(left_path)
                results['original_right'].append(right_path)
                
                # Update ImageItem with processed paths
                item.processed_images['original'] = {
                    'left': left_path,
                    'right': right_path
                }
                
                # Process masks if requested and available
                if process_masks and item.masks:
                    mask_results = self._process_masks_for_item(item, output_dir)
                    
                    # Merge mask results
                    for feature, paths in mask_results.items():
                        if feature not in results['mask_results']:
                            results['mask_results'][feature] = []
                        results['mask_results'][feature].append(paths)
                        
                        # Update ImageItem with mask processed paths
                        item.processed_images[feature] = paths
                
                logger.debug(f"Processed {item.image_id} ({i+1}/{len(items)})")
                
            except Exception as e:
                logger.error(f"Failed to process {item.image_id}: {e}")
                failed_items.append(item.image_id)
                
                # Add None entries to maintain list alignment
                results['original_left'].append(None)
                results['original_right'].append(None)
        
        success_count = len(items) - len(failed_items)
        logger.info(f"Batch processing completed: {success_count}/{len(items)} successful")
        
        if failed_items:
            logger.warning(f"Failed items: {failed_items}")
        
        return results
    
    def _process_original_image(self, item: ImageItem, output_dir: Path) -> Tuple[Path, Path]:
        """Process original image (split and downscale)"""
        if not item.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {item.image_path}")
        
        # Create output paths
        base_name = item.get_base_filename()
        left_path = output_dir / "original" / f"{base_name}_left.png"
        right_path = output_dir / "original" / f"{base_name}_right.png"
        
        # Ensure output directory exists
        left_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process image using existing downscale function
        try:
            downscale(
                str(item.image_path),
                str(left_path),
                str(right_path)
            )
            
            # Validate output files
            if not left_path.exists() or not right_path.exists():
                raise RuntimeError("Downscale operation failed to create output files")
            
            logger.debug(f"Processed original image: {item.image_id}")
            return left_path, right_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to process original image {item.image_id}: {e}")
    
    def _process_masks_for_item(self, item: ImageItem, output_dir: Path) -> Dict[str, Dict[str, Path]]:
        """Process all masks for an image item"""
        mask_results = {}
        
        for feature, mask_path in item.masks.items():
            if not mask_path or not mask_path.exists():
                logger.warning(f"Mask not found for {item.image_id}:{feature}")
                continue
            
            try:
                left_path, right_path = self._process_mask(item, feature, mask_path, output_dir)
                mask_results[feature] = {
                    'left': left_path,
                    'right': right_path
                }
                
            except Exception as e:
                logger.error(f"Failed to process mask {feature} for {item.image_id}: {e}")
                mask_results[feature] = {'left': None, 'right': None}
        
        return mask_results
    
    def _process_mask(self, item: ImageItem, feature: str, mask_path: Path, 
                     output_dir: Path) -> Tuple[Path, Path]:
        """Process a single mask (split and downscale)"""
        base_name = item.get_base_filename()
        left_path = output_dir / feature / f"{base_name}_left_mask.png"
        right_path = output_dir / feature / f"{base_name}_right_mask.png"
        
        # Ensure output directory exists
        left_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # For masks, we need to handle the processing differently
            # since they might be 1024x1024 instead of 2048x1024
            self._process_mask_file(mask_path, left_path, right_path)
            
            # Validate output files
            if not left_path.exists() or not right_path.exists():
                raise RuntimeError("Mask processing failed to create output files")
            
            return left_path, right_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to process mask {feature}: {e}")
    
    def _process_mask_file(self, input_path: Path, left_path: Path, right_path: Path):
        """Process mask file with appropriate handling for different sizes"""
        import cv2
        import numpy as np
        
        # Load mask
        mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask from {input_path}")
        
        height, width = mask.shape
        
        # Handle different mask sizes
        if width == 2048 and height == 1024:
            # Standard Cityscapes image size - split normally
            left_half = mask[:, :1024]
            right_half = mask[:, 1024:]
        elif width == 1024 and height == 1024:
            # Square mask - split in half
            left_half = mask[:, :512]
            right_half = mask[:, 512:]
        elif width == height:
            # Square mask of any size - split in half
            mid = width // 2
            left_half = mask[:, :mid]
            right_half = mask[:, mid:]
        else:
            # Unusual size - try to split in half width-wise
            mid = width // 2
            left_half = mask[:, :mid]
            right_half = mask[:, mid:]
            logger.warning(f"Unusual mask size {width}x{height}, splitting at midpoint")
        
        # Resize to target size
        left_resized = cv2.resize(left_half, (self.target_size, self.target_size), 
                                 interpolation=cv2.INTER_NEAREST)  # Use nearest for masks
        right_resized = cv2.resize(right_half, (self.target_size, self.target_size), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Save processed masks
        cv2.imwrite(str(left_path), left_resized)
        cv2.imwrite(str(right_path), right_resized)
    
    def process_single_image(self, item: ImageItem, output_dir: Path, 
                           process_masks: bool = True) -> Dict[str, any]:
        """
        Process a single image item.
        
        Args:
            item: ImageItem to process
            output_dir: Output directory
            process_masks: Whether to process masks
        
        Returns:
            Dictionary with processing results for this item
        """
        results = {
            'original': None,
            'masks': {}
        }
        
        try:
            # Process original image
            left_path, right_path = self._process_original_image(item, output_dir)
            results['original'] = {'left': left_path, 'right': right_path}
            
            # Update ImageItem
            item.processed_images['original'] = results['original']
            
            # Process masks if requested
            if process_masks and item.masks:
                mask_results = self._process_masks_for_item(item, output_dir)
                results['masks'] = mask_results
                
                # Update ImageItem
                for feature, paths in mask_results.items():
                    item.processed_images[feature] = paths
            
            logger.info(f"Successfully processed {item.image_id}")
            
        except Exception as e:
            logger.error(f"Failed to process {item.image_id}: {e}")
            raise
        
        return results
    
    def validate_processed_images(self, items: List[ImageItem]) -> Dict[str, any]:
        """
        Validate processed images for a batch of items.
        
        Args:
            items: List of ImageItem objects with processed image paths
        
        Returns:
            Validation report dictionary
        """
        validation_report = {
            'total_items': len(items),
            'items_with_valid_originals': 0,
            'items_with_missing_originals': 0,
            'mask_validation': {},
            'missing_files': [],
            'size_validation': {'correct_size': 0, 'incorrect_size': 0}
        }
        
        for item in items:
            # Validate original images
            original_paths = item.processed_images.get('original', {})
            left_path = original_paths.get('left')
            right_path = original_paths.get('right')
            
            if left_path and right_path and left_path.exists() and right_path.exists():
                validation_report['items_with_valid_originals'] += 1
                
                # Validate image sizes
                if self._validate_image_size(left_path) and self._validate_image_size(right_path):
                    validation_report['size_validation']['correct_size'] += 1
                else:
                    validation_report['size_validation']['incorrect_size'] += 1
            else:
                validation_report['items_with_missing_originals'] += 1
                if not left_path or not left_path.exists():
                    validation_report['missing_files'].append(f"{item.image_id}:original_left")
                if not right_path or not right_path.exists():
                    validation_report['missing_files'].append(f"{item.image_id}:original_right")
            
            # Validate mask images
            for feature, paths in item.processed_images.items():
                if feature == 'original':
                    continue
                
                if feature not in validation_report['mask_validation']:
                    validation_report['mask_validation'][feature] = {
                        'valid_items': 0, 'missing_items': 0
                    }
                
                left_path = paths.get('left')
                right_path = paths.get('right')
                
                if left_path and right_path and left_path.exists() and right_path.exists():
                    validation_report['mask_validation'][feature]['valid_items'] += 1
                else:
                    validation_report['mask_validation'][feature]['missing_items'] += 1
                    if not left_path or not left_path.exists():
                        validation_report['missing_files'].append(f"{item.image_id}:{feature}_left")
                    if not right_path or not right_path.exists():
                        validation_report['missing_files'].append(f"{item.image_id}:{feature}_right")
        
        # Calculate success rates
        validation_report['original_success_rate'] = (
            validation_report['items_with_valid_originals'] / validation_report['total_items'] * 100
            if validation_report['total_items'] > 0 else 0
        )
        
        return validation_report
    
    def _validate_image_size(self, image_path: Path) -> bool:
        """Validate that processed image has correct size"""
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            
            height, width = img.shape[:2]
            return width == self.target_size and height == self.target_size
            
        except Exception:
            return False
    
    def merge_processed_images(self, left_path: Path, right_path: Path, 
                             output_path: Path) -> Path:
        """
        Merge processed left and right images back into full-size image.
        
        Args:
            left_path: Path to left half image
            right_path: Path to right half image
            output_path: Path for merged output
        
        Returns:
            Path to merged image
        """
        try:
            # Use existing upscale function
            upscale(str(left_path), str(right_path), str(output_path))
            
            if not output_path.exists():
                raise RuntimeError("Merge operation failed to create output file")
            
            logger.debug(f"Merged images: {left_path.name} + {right_path.name} -> {output_path.name}")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to merge images: {e}")


if __name__ == "__main__":
    # Test image processor adapter
    import tempfile
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test processor
    processor = ImageProcessorAdapter(target_size=256)  # Smaller for testing
    
    print("ImageProcessorAdapter initialized successfully")
    print(f"Target size: {processor.target_size}x{processor.target_size}")
    print("Note: Full testing requires actual image files")
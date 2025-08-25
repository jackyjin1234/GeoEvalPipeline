#!/usr/bin/env python3
"""
Binary Mask Generator Adapter

This adapter integrates the existing createBinaryMasks.py functionality
with the optimized pipeline architecture, providing batch processing
and ordered execution capabilities.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

from utils.createBinaryMasks import (
    getSelectedLabels, createBinaryMask, processSingleFile
)
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.csHelpers import ensurePath

from core.pipeline_dataset import ImageItem

logger = logging.getLogger(__name__)


class MaskGeneratorAdapter:
    """
    Adapter for createBinaryMasks functionality optimized for pipeline processing.
    
    Provides batch processing capabilities and integration with pipeline data structures.
    """
    
    def __init__(self, target_features: List[str], target_labels: Optional[List[str]] = None,
                 combine_masks: bool = False, mask_suffix: str = "mask", dilate_masks: bool = True, 
                 kernel_size: int = 5, dilation_iterations: int = 5):
        """
        Initialize mask generator adapter.
        
        Args:
            target_features: List of feature names to generate masks for
            target_labels: Optional list of specific label names to use instead of categories
            combine_masks: Whether to create combined masks vs separate masks
            mask_suffix: Suffix for output mask files
            dilate_masks: Whether to dilate generated masks
            kernel_size: Kernel size for dilation
            dilation_iterations: Number of dilation iterations
        """
        self.target_features = target_features
        self.target_labels = target_labels
        self.combine_masks = combine_masks
        self.mask_suffix = mask_suffix
        self.dilate_masks = dilate_masks
        self.kernel_size = kernel_size
        self.dilation_iterations = dilation_iterations
        
        # Validate features
        self._validate_features()
        
        logger.info(f"MaskGeneratorAdapter initialized for features: {target_features}")
        if target_labels:
            logger.info(f"Using specific labels: {target_labels}")
        if dilate_masks:
            logger.info(f"Mask dilation enabled: kernel_size={kernel_size}, iterations={dilation_iterations}")
    
    def _validate_features(self):
        """Validate that requested features are available in Cityscapes labels"""
        try:
            # Test feature selection to ensure they're valid
            selected_labels = getSelectedLabels(
                categories=self.target_features, 
                label_names=self.target_labels
            )
            if not selected_labels:
                raise ValueError(f"No valid labels found for features: {self.target_features}")
            
            logger.info(f"Validated {len(selected_labels)} labels for requested features")
            
        except Exception as e:
            raise ValueError(f"Feature validation failed: {e}")
    
    def generate_masks_batch(self, items: List[ImageItem], output_dir: Path) -> Dict[str, List[Path]]:
        """
        Generate binary masks for a batch of image items.
        
        Args:
            items: List of ImageItem objects to process
            output_dir: Base output directory for masks
        
        Returns:
            Dictionary mapping feature names to lists of generated mask paths
        """
        ensurePath(str(output_dir))
        
        results = {feature: [] for feature in self.target_features}
        failed_items = []
        
        logger.info(f"Starting batch mask generation for {len(items)} items")
        
        for i, item in enumerate(items):
            try:
                item_masks = self._generate_masks_for_item(item, output_dir)
                
                # Update results
                for feature, mask_path in item_masks.items():
                    results[feature].append(mask_path)
                    
                    # Update ImageItem with mask path
                    item.masks[feature] = mask_path
                
                logger.debug(f"Generated masks for {item.image_id} ({i+1}/{len(items)})")
                
            except Exception as e:
                logger.error(f"Failed to generate masks for {item.image_id}: {e}")
                failed_items.append(item.image_id)
                
                # Add None entries to maintain list alignment
                for feature in self.target_features:
                    results[feature].append(None)
        
        success_count = len(items) - len(failed_items)
        logger.info(f"Batch mask generation completed: {success_count}/{len(items)} successful")
        
        if failed_items:
            logger.warning(f"Failed items: {failed_items}")
        
        return results
    
    def _generate_masks_for_item(self, item: ImageItem, output_dir: Path) -> Dict[str, Path]:
        """Generate masks for a single image item"""
        if not item.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {item.annotation_path}")
        
        # Load annotation
        annotation = Annotation()
        annotation.fromJsonFile(str(item.annotation_path))
        
        # Get base filename for output
        base_name = item.get_base_filename()
        
        # Create feature-specific output directories
        feature_masks = {}
        
        if self.combine_masks:
            # Single combined mask for all features
            combined_mask = self._create_combined_mask(annotation, self.target_features)
            
            mask_filename = f"{base_name}_{self.mask_suffix}.png"
            mask_path = output_dir / "combined" / mask_filename
            
            # Ensure directory exists
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            
            combined_mask.save(str(mask_path))
            
            # Map all features to the same combined mask
            for feature in self.target_features:
                feature_masks[feature] = mask_path
                
        else:
            # Separate masks for each feature
            for feature in self.target_features:
                feature_mask = self._create_feature_mask(annotation, feature)
                
                mask_filename = f"{base_name}_{feature}_{self.mask_suffix}.png"
                mask_path = output_dir / feature / mask_filename
                
                # Ensure directory exists
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                
                feature_mask.save(str(mask_path))
                feature_masks[feature] = mask_path
        
        # Apply dilation if enabled
        if self.dilate_masks:
            self._dilate_masks(feature_masks)
        
        return feature_masks
    
    def _create_combined_mask(self, annotation: Annotation, features: List[str]):
        """Create a combined mask for multiple features"""
        # Get selected labels for all features
        selected_labels = getSelectedLabels(
            categories=features, 
            label_names=self.target_labels
        )
        
        # Create combined binary mask
        masks = createBinaryMask(annotation, selected_labels, combine_masks=True)
        
        return masks['combined']
    
    def _create_feature_mask(self, annotation: Annotation, feature: str):
        """Create a mask for a single feature"""
        # Get selected labels for this feature
        selected_labels = getSelectedLabels(
            categories=[feature] if not self.target_labels else [], 
            label_names=self.target_labels
        )
        
        if not selected_labels:
            logger.warning(f"No labels found for feature: {feature}")
            # Return empty mask
            from PIL import Image
            return Image.new("L", (annotation.imgWidth, annotation.imgHeight), 0)
        
        # Create binary mask for this feature
        masks = createBinaryMask(annotation, selected_labels, combine_masks=True)
        
        return masks['combined']
    
    def generate_masks_for_features(self, item: ImageItem, features: List[str], 
                                  output_dir: Path) -> Dict[str, Path]:
        """
        Generate masks for specific features of an image item.
        
        Args:
            item: ImageItem to process
            features: List of features to generate masks for
            output_dir: Output directory for masks
        
        Returns:
            Dictionary mapping feature names to mask file paths
        """
        if not item.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {item.annotation_path}")
        
        # Load annotation
        annotation = Annotation()
        annotation.fromJsonFile(str(item.annotation_path))
        
        base_name = item.get_base_filename()
        feature_masks = {}
        
        for feature in features:
            try:
                # Create mask for this feature
                feature_mask = self._create_feature_mask(annotation, feature)
                
                # Save mask
                mask_filename = f"{base_name}_{feature}_{self.mask_suffix}.png"
                mask_path = output_dir / feature / mask_filename
                
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                feature_mask.save(str(mask_path))
                
                feature_masks[feature] = mask_path
                
                logger.debug(f"Generated {feature} mask for {item.image_id}")
                
            except Exception as e:
                logger.error(f"Failed to generate {feature} mask for {item.image_id}: {e}")
                feature_masks[feature] = None
        
        # Apply dilation if enabled
        if self.dilate_masks:
            self._dilate_masks(feature_masks)
        
        return feature_masks
    
    def _dilate_masks(self, feature_masks: Dict[str, Path]):
        """Apply dilation to generated masks"""
        for feature, mask_path in feature_masks.items():
            if mask_path and mask_path.exists():
                try:
                    self._dilate_single_mask(mask_path)
                    logger.debug(f"Dilated mask: {mask_path}")
                except Exception as e:
                    logger.warning(f"Failed to dilate mask {mask_path}: {e}")
    
    def _dilate_single_mask(self, mask_path: Path):
        """Dilate a single mask file"""
        # Load image
        image = cv2.imread(str(mask_path))
        if image is None:
            raise ValueError(f"Could not load mask from {mask_path}")
        
        # Create kernel and dilate
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=self.dilation_iterations)
        
        # Save dilated image back to same path
        cv2.imwrite(str(mask_path), dilated_image)
    
    def validate_masks(self, items: List[ImageItem]) -> Dict[str, any]:
        """
        Validate generated masks for a batch of items.
        
        Args:
            items: List of ImageItem objects with mask paths
        
        Returns:
            Validation report dictionary
        """
        validation_report = {
            'total_items': len(items),
            'items_with_all_masks': 0,
            'items_with_missing_masks': 0,
            'missing_masks': [],
            'mask_statistics': {feature: {'count': 0, 'total_pixels': 0} for feature in self.target_features}
        }
        
        for item in items:
            item_has_all_masks = True
            
            for feature in self.target_features:
                mask_path = item.masks.get(feature)
                
                if not mask_path or not mask_path.exists():
                    item_has_all_masks = False
                    validation_report['missing_masks'].append(f"{item.image_id}:{feature}")
                else:
                    # Validate mask file
                    try:
                        from PIL import Image
                        mask_img = Image.open(mask_path)
                        
                        # Count non-zero pixels
                        mask_array = mask_img.convert('L')
                        non_zero_pixels = sum(1 for pixel in mask_array.getdata() if pixel > 0)
                        
                        validation_report['mask_statistics'][feature]['count'] += 1
                        validation_report['mask_statistics'][feature]['total_pixels'] += non_zero_pixels
                        
                    except Exception as e:
                        logger.error(f"Failed to validate mask {mask_path}: {e}")
                        item_has_all_masks = False
                        validation_report['missing_masks'].append(f"{item.image_id}:{feature}:corrupt")
            
            if item_has_all_masks:
                validation_report['items_with_all_masks'] += 1
            else:
                validation_report['items_with_missing_masks'] += 1
        
        # Calculate success rate
        validation_report['success_rate'] = (
            validation_report['items_with_all_masks'] / validation_report['total_items'] * 100
            if validation_report['total_items'] > 0 else 0
        )
        
        return validation_report
    
    def get_mask_statistics(self, mask_paths: List[Path]) -> Dict[str, any]:
        """
        Get statistics about generated masks.
        
        Args:
            mask_paths: List of mask file paths
        
        Returns:
            Dictionary with mask statistics
        """
        stats = {
            'total_masks': len(mask_paths),
            'valid_masks': 0,
            'empty_masks': 0,
            'average_coverage': 0.0,
            'file_sizes': []
        }
        
        total_coverage = 0.0
        
        for mask_path in mask_paths:
            if not mask_path or not mask_path.exists():
                continue
            
            try:
                from PIL import Image
                import numpy as np
                
                # Load mask
                mask_img = Image.open(mask_path).convert('L')
                mask_array = np.array(mask_img)
                
                # Calculate coverage
                total_pixels = mask_array.size
                non_zero_pixels = np.count_nonzero(mask_array)
                coverage = (non_zero_pixels / total_pixels) * 100
                
                total_coverage += coverage
                stats['valid_masks'] += 1
                
                if non_zero_pixels == 0:
                    stats['empty_masks'] += 1
                
                # File size
                stats['file_sizes'].append(mask_path.stat().st_size)
                
            except Exception as e:
                logger.error(f"Failed to analyze mask {mask_path}: {e}")
        
        if stats['valid_masks'] > 0:
            stats['average_coverage'] = total_coverage / stats['valid_masks']
            stats['average_file_size'] = sum(stats['file_sizes']) / len(stats['file_sizes'])
        
        return stats


if __name__ == "__main__":
    # Test mask generator adapter
    import tempfile
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test items (mock)
    test_items = []
    
    # This would normally use real ImageItem objects with valid paths
    # For testing, we'd need actual Cityscapes annotation files
    
    print("MaskGeneratorAdapter test setup completed")
    print("Note: Full testing requires actual Cityscapes annotation files")
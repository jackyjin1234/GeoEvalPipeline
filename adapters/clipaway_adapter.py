#!/usr/bin/env python3
"""
CLIPAway Integration Adapter

This adapter integrates the CLIPAway inference functionality with the
optimized pipeline architecture, providing batch processing capabilities
and efficient model management.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any
import shutil

# Add CLIPAway directory to path
# clipaway_path = Path(__file__).parent.parent.parent / "CLIPAway"
# sys.path.insert(0, str(clipaway_path))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from diffusers import StableDiffusionInpaintPipeline
from omegaconf import OmegaConf

# CLIPAway imports
from CLIPAway.model.clip_away import CLIPAway
from CLIPAway.dataset.dataset import TestDataset
try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToPILImage
    from diffusers import StableDiffusionInpaintPipeline
    from omegaconf import OmegaConf
    
    # CLIPAway imports
    from CLIPAway.model.clip_away import CLIPAway
    from CLIPAway.dataset.dataset import TestDataset
    
    CLIPAWAY_AVAILABLE = True
except Exception as e:
    logging.getLogger(__name__).warning(f"CLIPAway dependencies not available: {e}")
    CLIPAWAY_AVAILABLE = False

from core.pipeline_dataset import ImageItem

logger = logging.getLogger(__name__)


class CLIPAwayAdapter:
    """
    Adapter for CLIPAway visual cue removal functionality.
    
    Manages model loading, batch processing, and integration with pipeline
    data structures for efficient visual cue removal operations.
    """
    
    def __init__(self, clipaway_path, device: str = "cuda", strength: float = 1.0, scale: int = 1,
                 seed: int = 42, model_key: str = "botp/stable-diffusion-v1-5-inpainting"):
        """
        Initialize CLIPAway adapter.
        
        Args:
            device: Device to use ("cuda" or "cpu")
            strength: CLIPAway processing strength
            scale: CLIPAway scale parameter
            seed: Random seed for reproducibility
            model_key: Stable Diffusion model key
        """
        if not CLIPAWAY_AVAILABLE:
            raise RuntimeError("CLIPAway dependencies not available")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.strength = strength
        self.scale = scale
        self.seed = seed
        self.model_key = model_key
        
        # Model components (loaded lazily)
        self.clipaway_model = None
        self.sd_pipeline = None
        self.latents = None
        
        # Configuration paths (will be set during initialization)
        self.clipaway_root = Path(clipaway_path)
        self._validate_clipaway_setup()
        
        logger.info(f"CLIPAwayAdapter initialized - Device: {self.device}")
    
    def _validate_clipaway_setup(self):
        """Validate CLIPAway installation and model availability"""
        required_paths = [
            self.clipaway_root / "ckpts" / "AlphaCLIP" / "clip_l14_grit+mim_fultune_6xe.pth",
            self.clipaway_root / "ckpts" / "IPAdapter" / "ip-adapter_sd15.bin",
            self.clipaway_root / "ckpts" / "CLIPAway" / "model.safetensors",
            self.clipaway_root / "ckpts" / "IPAdapter" / "image_encoder"
        ]
        
        missing_paths = [path for path in required_paths if not path.exists()]
        
        if missing_paths:
            logger.warning("Some CLIPAway model files not found:")
            for path in missing_paths:
                logger.warning(f"  Missing: {path}")
            logger.warning("Download models using: bash download_pretrained_models.sh")
    
    def _load_models(self):
        """Load CLIPAway models (lazy loading)"""
        if self.clipaway_model is not None:
            return  # Already loaded
        
        logger.info("Loading CLIPAway models...")
        
        try:
            # Load Stable Diffusion pipeline
            self.sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_key, 
                safety_checker=None, 
                torch_dtype=torch.float32
            )
            
            # Create CLIPAway model
            self.clipaway_model = CLIPAway(
                sd_pipe=self.sd_pipeline,
                image_encoder_path=str(self.clipaway_root / "ckpts" / "IPAdapter" / "image_encoder"),
                ip_ckpt=str(self.clipaway_root / "ckpts" / "IPAdapter" / "ip-adapter_sd15.bin"),
                alpha_clip_path=str(self.clipaway_root / "ckpts" / "AlphaCLIP" / "clip_l14_grit+mim_fultune_6xe.pth"),
                config=self._create_clipaway_config(),
                alpha_clip_id="ViT-L/14",
                device=self.device,
                num_tokens=4
            )
            
            # Prepare latents for consistent generation
            self.latents = torch.randn(
                (1, 4, 64, 64), 
                generator=torch.Generator().manual_seed(self.seed)
            ).to(self.device)
            
            logger.info("CLIPAway models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIPAway models: {e}")
            raise RuntimeError(f"CLIPAway model loading failed: {e}")
    
    def _create_clipaway_config(self) -> Any:
        """Create CLIPAway configuration object"""
        config_dict = {
            'device': self.device,
            'seed': self.seed,
            'scale': self.scale,
            'strength': self.strength,
            'number_of_hidden_layers': 6,
            'alpha_clip_embed_dim': 768,
            'ip_adapter_embed_dim': 1024,
            'mlp_projection_layer_ckpt_path': str(self.clipaway_root / "ckpts" / "CLIPAway" / "model.safetensors")
        }
        
        return OmegaConf.create(config_dict)
    
    def process_items_batch(self, items: List[ImageItem], features: List[str], 
                          output_dir: Path) -> Dict[str, List[Path]]:
        """
        Process a batch of items to remove visual cues using CLIPAway.
        
        Args:
            items: List of ImageItem objects with processed images and masks
            features: List of features to remove
            output_dir: Output directory for results
        
        Returns:
            Dictionary mapping features to lists of output image paths
        """
        # Load models if not already loaded
        self._load_models()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {feature: [] for feature in features}
        failed_items = []
        
        logger.info(f"Starting CLIPAway batch processing for {len(items)} items, {len(features)} features")
        
        for i, item in enumerate(items):
            try:
                item_results = self._process_item_features(item, features, output_dir)
                
                # Update results
                for feature, result_path in item_results.items():
                    results[feature].append(result_path)
                    
                    # Update ImageItem with CLIPAway result
                    item.clipaway_results[feature] = result_path
                
                logger.debug(f"Processed CLIPAway for {item.image_id} ({i+1}/{len(items)})")
                
            except Exception as e:
                logger.error(f"Failed CLIPAway processing for {item.image_id}: {e}")
                failed_items.append(item.image_id)
                
                # Add None entries to maintain list alignment
                for feature in features:
                    results[feature].append(None)
        
        success_count = len(items) - len(failed_items)
        logger.info(f"CLIPAway batch processing completed: {success_count}/{len(items)} successful")
        
        if failed_items:
            logger.warning(f"Failed items: {failed_items}")
        
        return results
    
    def _process_item_features(self, item: ImageItem, features: List[str], 
                             output_dir: Path) -> Dict[str, Path]:
        """Process all features for a single item"""
        item_results = {}
        
        for feature in features:
            try:
                result_path = self._process_single_feature(item, feature, output_dir)
                item_results[feature] = result_path
                
            except Exception as e:
                logger.error(f"Failed to process feature {feature} for {item.image_id}: {e}")
                item_results[feature] = None
        
        return item_results
    
    def _process_single_feature(self, item: ImageItem, feature: str, 
                              output_dir: Path) -> Path:
        """Process a single feature removal for an item"""
        # Check if we have processed images and masks for this feature
        original_images = item.processed_images.get('original', {})
        feature_masks = item.processed_images.get(feature, {})
        
        if not original_images or not feature_masks:
            raise ValueError(f"Missing processed images or masks for {feature}")
        
        # Create temporary dataset structure for CLIPAway
        temp_dataset_dir = self._create_temp_dataset(item, feature, original_images, feature_masks)
        
        try:
            # Process with CLIPAway
            result_path = self._run_clipaway_inference(temp_dataset_dir, item, feature, output_dir)
            return result_path
            
        finally:
            # Cleanup temporary dataset
            if temp_dataset_dir.exists():
                shutil.rmtree(temp_dataset_dir)
    
    def _create_temp_dataset(self, item: ImageItem, feature: str, 
                           original_images: Dict[str, Path], 
                           feature_masks: Dict[str, Path]) -> Path:
        """Create temporary dataset structure for CLIPAway"""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"clipaway_{item.image_id}_{feature}_"))
        
        images_dir = temp_dir / "images"
        masks_dir = temp_dir / "masks"
        
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        
        # Process both left and right sides
        left_image = original_images.get('left')
        left_mask = feature_masks.get('left')
        right_image = original_images.get('right')
        right_mask = feature_masks.get('right')
        
        if not left_image or not left_mask or not left_image.exists() or not left_mask.exists():
            raise ValueError(f"Missing left image or mask for {item.image_id}:{feature}")
        
        if not right_image or not right_mask or not right_image.exists() or not right_mask.exists():
            raise ValueError(f"Missing right image or mask for {item.image_id}:{feature}")
        
        # Copy to dataset structure with standard naming
        target_left_image = images_dir / f"{item.image_id}_left.jpg"
        target_left_mask = masks_dir / f"{item.image_id}_left.png"
        target_right_image = images_dir / f"{item.image_id}_right.jpg"
        target_right_mask = masks_dir / f"{item.image_id}_right.png"
        
        shutil.copy2(left_image, target_left_image)
        shutil.copy2(left_mask, target_left_mask)
        shutil.copy2(right_image, target_right_image)
        shutil.copy2(right_mask, target_right_mask)
        
        return temp_dir
    
    def _run_clipaway_inference(self, dataset_dir: Path, item: ImageItem, 
                              feature: str, output_dir: Path) -> Path:
        """Run CLIPAway inference on prepared dataset"""
        try:
            # Create test dataset
            test_dataset = TestDataset(str(dataset_dir))
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
            
            processed_halves = []
            
            # Process through CLIPAway - process both left and right halves
            for batch in test_dataloader:
                image, mask, image_paths = batch["image"], batch["mask"], batch["image_path"]
                image_pil = [ToPILImage()(img) for img in image]
                mask_pil = [ToPILImage()(img) for img in mask]
                
                # Generate result
                result_image = self.clipaway_model.generate(
                    prompt=[""],
                    scale=self.scale,
                    seed=self.seed,
                    pil_image=image_pil,
                    alpha=mask_pil,
                    strength=self.strength,
                    latents=self.latents
                )[0]
                
                processed_halves.append(result_image)
            
            if len(processed_halves) != 2:
                raise RuntimeError(f"Expected 2 processed halves, got {len(processed_halves)}")
            
            # Merge the processed left and right halves
            merged_image = self._merge_processed_halves(processed_halves[0], processed_halves[1])
            
            # Save merged result
            result_filename = f"{item.image_id}_{feature}_removed.jpg"
            result_path = output_dir / feature / result_filename
            result_path.parent.mkdir(parents=True, exist_ok=True)
            
            merged_image.save(str(result_path))
            
            logger.debug(f"CLIPAway generated merged result: {result_path}")
            return result_path
            
        except Exception as e:
            raise RuntimeError(f"CLIPAway inference failed: {e}")
    
    def _merge_processed_halves(self, left_image, right_image):
        """Merge processed left and right image halves back into full image"""
        from PIL import Image
        
        # Get dimensions
        left_width, left_height = left_image.size
        right_width, right_height = right_image.size
        
        # Ensure both halves have same height
        if left_height != right_height:
            raise ValueError(f"Height mismatch: left={left_height}, right={right_height}")
        
        # Create merged image
        total_width = left_width + right_width
        merged_image = Image.new('RGB', (total_width, left_height))
        
        # Paste left and right halves
        merged_image.paste(left_image, (0, 0))
        merged_image.paste(right_image, (left_width, 0))
        
        return merged_image
    
    def process_single_item(self, item: ImageItem, feature: str, 
                          output_dir: Path) -> Optional[Path]:
        """
        Process a single item for a single feature.
        
        Args:
            item: ImageItem to process
            feature: Feature to remove
            output_dir: Output directory
        
        Returns:
            Path to generated image or None if failed
        """
        # Load models if needed
        self._load_models()
        
        try:
            result_path = self._process_single_feature(item, feature, output_dir)
            
            # Update ImageItem
            item.clipaway_results[feature] = result_path
            
            logger.info(f"CLIPAway processed {item.image_id}:{feature}")
            return result_path
            
        except Exception as e:
            logger.error(f"CLIPAway failed for {item.image_id}:{feature}: {e}")
            return None
    
    def validate_results(self, items: List[ImageItem], features: List[str]) -> Dict[str, Any]:
        """
        Validate CLIPAway processing results.
        
        Args:
            items: List of processed ImageItem objects
            features: List of features that should have been processed
        
        Returns:
            Validation report dictionary
        """
        validation_report = {
            'total_items': len(items),
            'total_features': len(features),
            'successful_results': 0,
            'failed_results': 0,
            'missing_results': [],
            'feature_statistics': {feature: {'success': 0, 'failure': 0} for feature in features}
        }
        
        for item in items:
            for feature in features:
                result_path = item.clipaway_results.get(feature)
                
                if result_path and result_path.exists():
                    validation_report['successful_results'] += 1
                    validation_report['feature_statistics'][feature]['success'] += 1
                    
                    # Validate image file
                    if not self._validate_result_image(result_path):
                        logger.warning(f"Invalid result image: {result_path}")
                        validation_report['failed_results'] += 1
                        validation_report['feature_statistics'][feature]['failure'] += 1
                        
                else:
                    validation_report['failed_results'] += 1
                    validation_report['feature_statistics'][feature]['failure'] += 1
                    validation_report['missing_results'].append(f"{item.image_id}:{feature}")
        
        # Calculate success rate
        total_expected = validation_report['total_items'] * validation_report['total_features']
        validation_report['success_rate'] = (
            validation_report['successful_results'] / total_expected * 100
            if total_expected > 0 else 0
        )
        
        return validation_report
    
    def _validate_result_image(self, image_path: Path) -> bool:
        """Validate that result image is valid"""
        try:
            from PIL import Image
            img = Image.open(image_path)
            
            # Basic validation - ensure it's a valid image
            img.verify()
            
            # Check size is reasonable
            width, height = img.size
            if width < 100 or height < 100:  # Too small
                return False
            
            return True
            
        except Exception:
            return False
    
    def cleanup_models(self):
        """Clean up loaded models to free memory"""
        if self.clipaway_model is not None:
            del self.clipaway_model
            self.clipaway_model = None
        
        if self.sd_pipeline is not None:
            del self.sd_pipeline
            self.sd_pipeline = None
        
        if self.latents is not None:
            del self.latents
            self.latents = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("CLIPAway models cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'models_loaded': self.clipaway_model is not None,
            'device': self.device,
            'strength': self.strength,
            'scale': self.scale,
            'seed': self.seed,
            'model_key': self.model_key
        }
        
        if torch.cuda.is_available() and self.device == "cuda":
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)
        
        return info


if __name__ == "__main__":
    # Test CLIPAway adapter
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if not CLIPAWAY_AVAILABLE:
        print("CLIPAway dependencies not available - skipping test")
        sys.exit(0)
    
    try:
        # Create test adapter
        adapter = CLIPAwayAdapter(device="cpu")  # Use CPU for testing
        
        print("CLIPAwayAdapter initialized successfully")
        print("Model info:", adapter.get_model_info())
        
        # Test model loading (if models are available)
        try:
            adapter._load_models()
            print("Models loaded successfully")
        except Exception as e:
            print(f"Note: Model loading failed (expected if models not downloaded): {e}")
        
    except Exception as e:
        print(f"Adapter initialization failed: {e}")
    
    print("CLIPAway adapter test completed")
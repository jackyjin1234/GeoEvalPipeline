#!/usr/bin/env python3
"""
Geolocation Evaluator Adapter

This adapter integrates the geolocation evaluation functionality with the
optimized pipeline architecture, providing batch processing and systematic
evaluation of visual cue importance for geolocation tasks.
"""

import logging
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Add parent directory to path to import existing evaluation tools
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# try:
from utils.geoLocationAPI import GeoLocationAPI
from utils.distanceCalculator import DistanceCalculator
from utils.resultsDatabase import ResultsDatabase
EVALUATION_AVAILABLE = True
# except ImportError as e:
#     logging.getLogger(__name__).warning(f"Evaluation dependencies not available: {e}")
#     EVALUATION_AVAILABLE = False

from core.pipeline_dataset import ImageItem

logger = logging.getLogger(__name__)


# class MockGeoLocationAPI:
#     """Mock API for testing without actual OpenAI calls"""
    
#     def __init__(self, *args, **kwargs):
#         pass
    
#     async def get_geolocation_prediction(self, image_path: str) -> Optional[Dict]:
#         """Mock geolocation prediction"""
#         import random
#         import time
        
#         # Simulate processing time
#         await asyncio.sleep(0.1)
        
#         # Return mock prediction
#         return {
#             'coordinates': (50.0 + random.uniform(-5, 5), 6.0 + random.uniform(-5, 5)),
#             'confidence': random.choice(['High', 'Medium', 'Low']),
#             'region': 'Mock City, Mock Country',
#             'reasoning': 'Mock reasoning for testing purposes',
#             'critical_features': 'Mock critical features',
#             'processing_time': 0.1,
#             'model_used': 'mock-model',
#             'raw_response': 'Mock raw response'
#         }


class EvaluatorAdapter:
    """
    Adapter for geolocation evaluation functionality optimized for pipeline processing.
    
    Handles systematic evaluation of visual cue importance by comparing
    geolocation accuracy between original and feature-removed images.
    """
    
    def __init__(self, model: str = "gpt-4-vision-preview", concurrent_requests: int = 3,
                 cache_responses: bool = True, use_ground_truth: bool = True,
                 mock_mode: bool = False):
        """
        Initialize evaluator adapter.
        
        Args:
            model: OpenAI model to use for geolocation
            concurrent_requests: Number of concurrent API requests
            cache_responses: Whether to cache API responses
            use_ground_truth: Whether to use actual GPS coordinates
            mock_mode: Use mock API for testing
        """
        self.model = model
        self.concurrent_requests = concurrent_requests
        self.cache_responses = cache_responses
        self.use_ground_truth = use_ground_truth
        self.mock_mode = mock_mode
        
        # Initialize components
        # if mock_mode or not EVALUATION_AVAILABLE:
        #     self.api = MockGeoLocationAPI()
        #     logger.info("Using mock geolocation API")
        # else:
        self.api = GeoLocationAPI(
            model=model,
            concurrent_requests=concurrent_requests,
            cache_responses=cache_responses
        )
        logger.info(f"Using OpenAI geolocation API - Model: {model}")
        
        if EVALUATION_AVAILABLE:
            self.distance_calc = DistanceCalculator()
        else:
            self.distance_calc = None
            
        self.database = None  # Will be initialized when needed
        
        logger.info("EvaluatorAdapter initialized")
    
    def _calculate_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """Calculate distance between coordinates"""
        if self.distance_calc:
            return self.distance_calc.haversine_distance(coord1, coord2)
        else:
            # Simple distance calculation for mock mode
            import math
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111.0  # Rough km conversion
    
    async def evaluate_items_batch(self, items: List[ImageItem], features: List[str],
                                 output_dir: Path) -> Dict[str, Any]:
        """
        Evaluate geolocation accuracy for a batch of items.
        
        Args:
            items: List of ImageItem objects with original and processed images
            features: List of features that were removed
            output_dir: Output directory for evaluation results
        
        Returns:
            Dictionary with evaluation results and statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        db_path = output_dir / "evaluation_results.db"
        if EVALUATION_AVAILABLE:
            self.database = ResultsDatabase(str(db_path))
        
        # Simple results tracking
        results = {}
        
        logger.info(f"Starting batch evaluation for {len(items)} items, {len(features)} features")
        
        # Process items with controlled concurrency
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        
        async def process_item_with_semaphore(item):
            async with semaphore:
                return await self._evaluate_single_item(item, features)
        
        # Create tasks for all items
        tasks = [process_item_with_semaphore(item) for item in items]
        
        # Process with progress tracking
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                item_results = await task
                
                # Update results - simplified structure
                item_id = item_results['item_id']
                results[item_id] = {
                    'ground_truth_coords': item_results['ground_truth_coords'],
                    'feature_results': item_results['feature_results']
                }
                
                # Skip database storage to keep it simple
                
                logger.debug(f"Evaluated item {i+1}/{len(items)}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate item: {e}")
        
        # Save results to files
        await self._save_evaluation_results(results, output_dir)
        
        logger.info(f"Batch evaluation completed: {len(results)} items evaluated")
        return {'completed_evaluations': len(results) * (1 + len(features))}
    
    async def _evaluate_single_item(self, item: ImageItem, features: List[str]) -> Dict[str, Any]:
        """Evaluate a single item for all features"""
        item_results = {
            'item_id': item.image_id,
            'ground_truth_coords': item.ground_truth_coords,
            'feature_results': []
        }
        
        # Evaluate original image
        try:
            original_result = await self._evaluate_original_image(item)
            item_results['feature_results'].append(original_result)
            
        except Exception as e:
            logger.exception(f"Failed to evaluate original image for {item.image_id}")
            original_result = None
        
        # Evaluate feature-removed images
        for feature in features:
            try:
                feature_result = await self._evaluate_feature_removed_image(
                    item, feature, original_result
                )
                item_results['feature_results'].append(feature_result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {feature} for {item.image_id}: {e}")
        
        return item_results
    
    async def _evaluate_original_image(self, item: ImageItem) -> Dict[str, Any]:
        """Evaluate original image geolocation accuracy"""
        # Use merged processed image or original image
        image_path = self._get_best_image_path(item, 'original')
        
        if not image_path or not image_path.exists():
            raise FileNotFoundError(f"Original image not found for {item.image_id}")
        
        # Get geolocation prediction
        prediction = await self.api.get_geolocation_prediction(str(image_path))
        
        if not prediction:
            raise RuntimeError(f"Failed to get prediction for {item.image_id}")
        
        # Calculate error if ground truth available
        error_km = None
        if self.use_ground_truth and item.ground_truth_coords and prediction.get('location_estimate'):
            error_km = self._calculate_distance(
                item.ground_truth_coords,
                (prediction['location_estimate']['latitude'], prediction['location_estimate']['longitude'])
            )
        
        return {
            'feature_removed': 'none',
            'error_km': error_km,
            'gpt_response': prediction['location_estimate'],
            'error_increase_km': 0.0
        }
    
    async def _evaluate_feature_removed_image(self, item: ImageItem, feature: str, 
                                            original_result: Optional[Dict]) -> Dict[str, Any]:
        """Evaluate feature-removed image geolocation accuracy"""
        # Get CLIPAway result image
        clipaway_path = item.clipaway_results.get(feature)
        
        if not clipaway_path or not clipaway_path.exists():
            raise FileNotFoundError(f"CLIPAway result not found for {item.image_id}:{feature}")
        
        # Get geolocation prediction
        prediction = await self.api.get_geolocation_prediction(str(clipaway_path))
        
        if not prediction:
            raise RuntimeError(f"Failed to get prediction for {item.image_id}:{feature}")
        
        # Calculate error if ground truth available
        error_km = None
        error_increase_km = None
        original_error_km = original_result.get('error_km') if original_result else None
        
        if self.use_ground_truth and item.ground_truth_coords and prediction.get('location_estimate'):
            error_km = self._calculate_distance(
                item.ground_truth_coords,
                (prediction['location_estimate']['latitude'], prediction['location_estimate']['longitude'])
            )
            
            if original_error_km is not None:
                error_increase_km = error_km - original_error_km
        
        return {
            'feature_removed': feature,
            'error_km': error_km,
            'gpt_response': prediction['location_estimate'],
            'error_increase_km': error_increase_km
        }
    
    def _get_best_image_path(self, item: ImageItem, image_type: str) -> Optional[Path]:
        """Get the best available image path for evaluation"""
        if image_type == 'original':
            # Prefer processed/merged image, fallback to original
            processed = item.processed_images.get('original', {})
            if processed.get('left') and processed.get('right'):
                # Merge left and right halves for evaluation
                return self._merge_original_halves(item, processed)
            else:
                return item.image_path
        else:
            # For feature-removed images, use CLIPAway results (already merged)
            return item.clipaway_results.get(image_type)
    
    def _merge_original_halves(self, item: ImageItem, processed: Dict[str, Path]) -> Optional[Path]:
        """Merge original left and right halves for evaluation"""
        try:
            from adapters.image_processor import ImageProcessorAdapter
            
            # Create temporary merged image
            output_dir = Path("/tmp") / "merged_evaluation"
            output_dir.mkdir(exist_ok=True)
            
            merged_filename = f"{item.image_id}_original_merged.jpg"
            merged_path = output_dir / merged_filename
            
            # Use image processor to merge
            processor = ImageProcessorAdapter()
            result_path = processor.merge_processed_images(
                processed['left'], 
                processed['right'], 
                merged_path
            )
            
            return result_path
            
        except Exception as e:
            logger.warning(f"Failed to merge original halves for {item.image_id}: {e}")
            # Fallback to left half only
            return processed.get('left')
    
    async def _save_evaluation_results(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation results to files"""
        import json
        
        # Save simplified results as JSON
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {results_file}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    
    def validate_evaluation_setup(self) -> Dict[str, Any]:
        """Validate evaluation setup and dependencies"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'api_available': not self.mock_mode,
            'database_available': EVALUATION_AVAILABLE,
            'distance_calc_available': self.distance_calc is not None
        }
        
        if self.mock_mode:
            validation['warnings'].append("Using mock API - results are not real")
        
        if not EVALUATION_AVAILABLE:
            validation['warnings'].append("Evaluation dependencies not fully available")
        
        if not self.use_ground_truth:
            validation['warnings'].append("Ground truth GPS coordinates disabled")
        
        return validation


if __name__ == "__main__":
    # Test evaluator adapter
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test adapter with mock mode
    adapter = EvaluatorAdapter(mock_mode=True)
    
    print("EvaluatorAdapter initialized successfully")
    
    # Test validation
    validation = adapter.validate_evaluation_setup()
    print("Setup validation:", validation)
    
    print("Evaluator adapter test completed")
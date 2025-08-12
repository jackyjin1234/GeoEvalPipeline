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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from pipeline.utils.geoLocationAPI import GeoLocationAPI
    from pipeline.utils.distanceCalculator import DistanceCalculator
    from pipeline.utils.resultsDatabase import ResultsDatabase
    EVALUATION_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Evaluation dependencies not available: {e}")
    EVALUATION_AVAILABLE = False

from core.pipeline_dataset import ImageItem

logger = logging.getLogger(__name__)


class MockGeoLocationAPI:
    """Mock API for testing without actual OpenAI calls"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    async def get_geolocation_prediction(self, image_path: str) -> Optional[Dict]:
        """Mock geolocation prediction"""
        import random
        import time
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Return mock prediction
        return {
            'coordinates': (50.0 + random.uniform(-5, 5), 6.0 + random.uniform(-5, 5)),
            'confidence': random.choice(['High', 'Medium', 'Low']),
            'region': 'Mock City, Mock Country',
            'reasoning': 'Mock reasoning for testing purposes',
            'critical_features': 'Mock critical features',
            'processing_time': 0.1,
            'model_used': 'mock-model',
            'raw_response': 'Mock raw response'
        }


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
        if mock_mode or not EVALUATION_AVAILABLE:
            self.api = MockGeoLocationAPI()
            logger.info("Using mock geolocation API")
        else:
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
        
        results = {
            'total_items': len(items),
            'total_evaluations': len(items) * (1 + len(features)),  # Original + features
            'completed_evaluations': 0,
            'failed_evaluations': 0,
            'results_by_item': {},
            'results_by_feature': {feature: [] for feature in features},
            'original_results': [],
            'summary_statistics': {}
        }
        
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
                
                # Update results
                item_id = item_results['item_id']
                results['results_by_item'][item_id] = item_results
                
                # Organize by feature
                for feature_result in item_results['feature_results']:
                    feature = feature_result['feature_removed']
                    if feature == 'none':
                        results['original_results'].append(feature_result)
                    else:
                        results['results_by_feature'][feature].append(feature_result)
                
                # Store in database if available
                if self.database:
                    for feature_result in item_results['feature_results']:
                        self.database.insert_result(feature_result)
                
                results['completed_evaluations'] += len(item_results['feature_results'])
                
                logger.debug(f"Evaluated item {i+1}/{len(items)}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate item: {e}")
                results['failed_evaluations'] += 1
        
        # Generate summary statistics
        results['summary_statistics'] = self._calculate_summary_statistics(results)
        
        # Save results to files
        await self._save_evaluation_results(results, output_dir)
        
        logger.info(f"Batch evaluation completed: {results['completed_evaluations']} evaluations")
        return results
    
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
            logger.error(f"Failed to evaluate original image for {item.image_id}: {e}")
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
        if self.use_ground_truth and item.ground_truth_coords and prediction.get('coordinates'):
            error_km = self._calculate_distance(
                item.ground_truth_coords,
                (prediction['location_estimate']['latitude'], prediction['location_estimate']['longtidute'])
            )
        
        return {
            'image_id': item.image_id,
            'feature_removed': 'none',
            'original_error_km': error_km,
            'modified_error_km': error_km,
            'error_increase_km': 0.0,
            'gpt_response': prediction['location_estimate'],
            'processing_time': prediction.get('processing_time', 0)
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
        modified_error_km = None
        error_increase_km = None
        original_error_km = original_result.get('original_error_km') if original_result else None
        
        if self.use_ground_truth and item.ground_truth_coords and prediction.get('coordinates'):
            modified_error_km = self._calculate_distance(
                item.ground_truth_coords,
                (prediction['location_estimate']['latitude'], prediction['location_estimate']['longtidute'])
            )
            
            if original_error_km is not None:
                error_increase_km = modified_error_km - original_error_km
        
        return {
            'image_id': item.image_id,
            'feature_removed': feature,
            'original_error_km': original_error_km,
            'modified_error_km': modified_error_km,
            'error_increase_km': error_increase_km,
            'gpt_response': prediction['location_estimate'],
            'processing_time': prediction.get('processing_time', 0)
        }
    
    def _get_best_image_path(self, item: ImageItem, image_type: str) -> Optional[Path]:
        """Get the best available image path for evaluation"""
        if image_type == 'original':
            # Prefer processed/merged image, fallback to original
            processed = item.processed_images.get('original', {})
            if processed.get('left') and processed.get('right'):
                # TODO: Merge left and right if needed
                # For now, use left side
                return processed.get('left')
            else:
                return item.image_path
        else:
            # For feature-removed images, use CLIPAway results
            return item.clipaway_results.get(image_type)
    
    def _calculate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results"""
        summary = {
            'overall': {
                'total_evaluations': results['completed_evaluations'],
                'success_rate': 0.0,
                'average_processing_time': 0.0
            },
            'by_feature': {}
        }
        
        # Original image statistics
        original_errors = [
            r['original_error_km'] for r in results['original_results'] 
            if r.get('original_error_km') is not None
        ]
        
        if original_errors:
            summary['original'] = {
                'count': len(original_errors),
                'average_error_km': sum(original_errors) / len(original_errors),
                'median_error_km': sorted(original_errors)[len(original_errors) // 2],
                'min_error_km': min(original_errors),
                'max_error_km': max(original_errors)
            }
        
        # Feature-specific statistics
        for feature, feature_results in results['results_by_feature'].items():
            error_increases = [
                r['error_increase_km'] for r in feature_results 
                if r.get('error_increase_km') is not None
            ]
            
            if error_increases:
                summary['by_feature'][feature] = {
                    'count': len(error_increases),
                    'average_error_increase_km': sum(error_increases) / len(error_increases),
                    'median_error_increase_km': sorted(error_increases)[len(error_increases) // 2],
                    'max_error_increase_km': max(error_increases),
                    'min_error_increase_km': min(error_increases)
                }
        
        # Calculate overall success rate
        total_predictions = sum(
            len(feature_results) for feature_results in results['results_by_feature'].values()
        ) + len(results['original_results'])
        
        successful_predictions = sum(
            1 for item_results in results['results_by_item'].values()
            for feature_result in item_results['feature_results']
            if feature_result.get('gpt_response') is not None
        )
        
        summary['overall']['success_rate'] = (
            successful_predictions / total_predictions * 100 
            if total_predictions > 0 else 0
        )
        
        return summary
    
    async def _save_evaluation_results(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation results to files"""
        import json
        
        # Save complete results as JSON
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = output_dir / "evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary_report(results))
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
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
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        summary = results['summary_statistics']
        
        report_lines = [
            "=" * 60,
            "GEOLOCATION EVALUATION SUMMARY",
            "=" * 60,
            f"Total items evaluated: {results['total_items']}",
            f"Total evaluations: {results['completed_evaluations']}",
            f"Failed evaluations: {results['failed_evaluations']}",
            f"Success rate: {summary['overall']['success_rate']:.1f}%",
            ""
        ]
        
        # Original images performance
        if 'original' in summary:
            orig_stats = summary['original']
            report_lines.extend([
                "ORIGINAL IMAGES:",
                f"  Average error: {orig_stats['average_error_km']:.2f} km",
                f"  Median error: {orig_stats['median_error_km']:.2f} km",
                f"  Error range: {orig_stats['min_error_km']:.2f} - {orig_stats['max_error_km']:.2f} km",
                ""
            ])
        
        # Feature impact analysis
        if summary['by_feature']:
            report_lines.append("FEATURE IMPACT ANALYSIS:")
            
            # Sort features by impact
            features_by_impact = sorted(
                summary['by_feature'].items(),
                key=lambda x: x[1]['average_error_increase_km'],
                reverse=True
            )
            
            for feature, stats in features_by_impact:
                report_lines.extend([
                    f"  {feature.upper()}:",
                    f"    Average error increase: {stats['average_error_increase_km']:.2f} km",
                    f"    Median error increase: {stats['median_error_increase_km']:.2f} km",
                    f"    Max error increase: {stats['max_error_increase_km']:.2f} km",
                    f"    Impact severity: {stats['impact_severity']}",
                    ""
                ])
        
        report_lines.extend([
            "=" * 60,
            f"Evaluation completed at: {__import__('datetime').datetime.now()}",
            "=" * 60
        ])
        
        return "\n".join(report_lines)
    
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
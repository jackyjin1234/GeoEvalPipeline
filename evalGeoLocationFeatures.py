#!/usr/bin/python
#
# Geolocation Feature Importance Evaluation Pipeline
# 
# This script evaluates how removing specific visual features affects GPT's
# geolocation accuracy on street view images.
#
# Usage: evalGeoLocationFeatures.py [OPTIONS] <input_directory> <results_database>
# Options:
#   -h, --help           Show help message
#   -f, --features       Comma-separated list of features to test (person,vehicle,object,all)
#   -m, --model          OpenAI model to use (default: gpt-4-vision-preview)
#   -c, --concurrent     Number of concurrent API calls (default: 5)
#   --original-only      Only evaluate original images (skip feature removal)
#   --cache-responses    Cache GPT responses to avoid redundant API calls
#   --max-images         Maximum number of images to process (for testing)
#
# Examples:
#   # Evaluate impact of removing people and vehicles
#   evalGeoLocationFeatures.py -f "person,vehicle" ./sample/ results.db
#   
#   # Full evaluation with all features
#   evalGeoLocationFeatures.py -f "all" ./dataset/ comprehensive_results.db

from __future__ import print_function, absolute_import, division
import os, sys, argparse, json, time
import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError, ensurePath
from cityscapesscripts.helpers.labels import name2label

# Local imports (will be created)
from .geoLocationAPI import GeoLocationAPI
from .distanceCalculator import DistanceCalculator
from .resultsDatabase import ResultsDatabase

class GeoLocationFeatureEvaluator:
    """Main class for evaluating geolocation feature importance"""
    
    SUPPORTED_FEATURES = ['human', 'vehicle', 'object', 'construction', 'nature']
    
    def __init__(self, args):
        self.args = args
        self.api = GeoLocationAPI(
            model=args.model,
            concurrent_requests=args.concurrent,
            cache_responses=args.cache_responses
        )
        self.distance_calc = DistanceCalculator()
        self.db = ResultsDatabase(args.results_database)
        self.processed_count = 0
        self.total_images = 0
        
    def get_features_to_test(self):
        """Parse feature arguments and return list of features to test"""
        if self.args.features.lower() == 'all':
            return self.SUPPORTED_FEATURES
        
        features = [f.strip() for f in self.args.features.split(',')]
        invalid_features = [f for f in features if f not in self.SUPPORTED_FEATURES]
        
        if invalid_features:
            printError(f"Invalid features: {invalid_features}. Supported: {self.SUPPORTED_FEATURES}")
        
        return features
    
    def find_image_sets(self, input_directory):
        """
        Find all image sets with original images and their feature-removed variants
        
        Expected structure:
        input_directory/
        ├── images/           # Original images
        ├── masks/            # Binary masks for each feature
        └── results/          # Feature-removed images from CLIPAway
        """
        input_path = Path(input_directory)
        images_dir = input_path / 'images'
        results_dir = input_path / 'results'
        
        if not images_dir.exists():
            printError(f"Images directory not found: {images_dir}")
        
        image_sets = []
        
        # Find all original images
        for img_file in images_dir.glob('*.jpg'):
            base_name = img_file.stem
            
            image_set = {
                'base_name': base_name,
                'original_image': str(img_file),
                'feature_removed_images': {}
            }
            
            # Look for feature-removed variants
            if results_dir.exists():
                for feature in self.get_features_to_test():
                    # Look for images with feature removed
                    feature_removed_pattern = f"{base_name}*{feature}*"
                    feature_files = list(results_dir.glob(feature_removed_pattern))
                    
                    if feature_files:
                        # Use the first match (could be enhanced to handle multiple)
                        image_set['feature_removed_images'][feature] = str(feature_files[0])
            
            image_sets.append(image_set)
        
        return image_sets[:self.args.max_images] if self.args.max_images else image_sets
    
    async def evaluate_single_image_set(self, image_set):
        """Evaluate geolocation accuracy for a single image set"""
        base_name = image_set['base_name']
        results = []
        
        try:
            # Get ground truth coordinates (if available in metadata)
            ground_truth = self.extract_ground_truth(image_set['original_image'])
            
            # Evaluate original image
            print(f"Processing original: {base_name}")
            original_prediction = await self.api.get_geolocation_prediction(
                image_set['original_image']
            )
            
            original_error = None
            if ground_truth and original_prediction:
                original_error = self.distance_calc.haversine_distance(
                    ground_truth, (original_prediction['location_estimate']['latitude'], original_prediction['location_estimate']['longtidute'])
                )
            
            # Store original result
            results.append({
                'image_id': base_name,
                'feature_removed': 'none',
                'original_error_km': original_error,
                'modified_error_km': original_error,
                'error_increase_km': 0.0,
                'gpt_response': original_prediction['location_estimate'],
                'processing_time': original_prediction.get('processing_time', 0) if original_prediction else 0
            })
            
            if not self.args.original_only:
                # Evaluate feature-removed variants
                for feature, feature_image_path in image_set['feature_removed_images'].items():
                    print(f"Processing {feature} removed: {base_name}")
                    
                    modified_prediction = await self.api.get_geolocation_prediction(
                        feature_image_path
                    )
                    
                    modified_error = None
                    error_increase = None
                    
                    if ground_truth and modified_prediction:
                        modified_error = self.distance_calc.haversine_distance(
                            ground_truth, (modified_prediction['location_estimate']['latitude'], modified_prediction['location_estimate']['longtidute'])
                        )
                        
                        if original_error is not None:
                            error_increase = modified_error - original_error
                    
                    results.append({
                        'image_id': base_name,
                        'feature_removed': feature,
                        'original_error_km': original_error,
                        'modified_error_km': modified_error,
                        'error_increase_km': error_increase,
                        'gpt_response': modified_prediction['location_estimate'],
                        'processing_time': modified_prediction.get('processing_time', 0) if modified_prediction else 0
                    })
            
            return results
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            return []
    
    def extract_ground_truth(self, image_path):
        """
        Extract ground truth coordinates from image metadata or filename
        This is a placeholder - implement based on your ground truth data format
        """
        # TODO: Implement based on how ground truth coordinates are stored
        # Options:
        # 1. EXIF GPS data in images
        # 2. Separate metadata JSON files
        # 3. Database lookup by filename
        # 4. Cityscapes GPS coordinates if available
        
        return None  # Placeholder
    
    async def run_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("Starting Geolocation Feature Importance Evaluation")
        print(f"Input directory: {self.args.input_directory}")
        print(f"Features to test: {', '.join(self.get_features_to_test())}")
        print(f"Model: {self.args.model}")
        print(f"Concurrent requests: {self.args.concurrent}")
        
        # Find all image sets
        image_sets = self.find_image_sets(self.args.input_directory)
        self.total_images = len(image_sets)
        
        if not image_sets:
            printError("No image sets found in input directory")
        
        print(f"Found {self.total_images} image sets to process")
        
        # Process image sets with concurrency control
        semaphore = asyncio.Semaphore(self.args.concurrent)
        
        async def process_with_semaphore(image_set):
            async with semaphore:
                return await self.evaluate_single_image_set(image_set)
        
        # Create tasks for all image sets
        tasks = [process_with_semaphore(image_set) for image_set in image_sets]
        
        # Process with progress tracking
        start_time = time.time()
        all_results = []
        
        for completed_task in asyncio.as_completed(tasks):
            results = await completed_task
            all_results.extend(results)
            self.processed_count += 1
            
            # Progress update
            progress = (self.processed_count / self.total_images) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / self.processed_count) * (self.total_images - self.processed_count)
            
            print(f"\rProgress: {progress:.1f}% ({self.processed_count}/{self.total_images}) "
                  f"ETA: {eta/60:.1f}min", end='', flush=True)
        
        print("\nSaving results to database...")
        
        # Save all results to database
        for result in all_results:
            self.db.insert_result(result)
        
        # Generate summary statistics
        self.generate_summary()
        
        print(f"\nEvaluation completed! Results saved to: {self.args.results_database}")
    
    def generate_summary(self):
        """Generate and display summary statistics"""
        summary = self.db.get_summary_statistics()
        
        print("\n" + "="*60)
        print("GEOLOCATION FEATURE IMPORTANCE SUMMARY")
        print("="*60)
        
        for feature, stats in summary.items():
            if feature == 'none':
                print(f"\nOriginal Images:")
                print(f"  Average error: {stats['avg_error']:.2f} km")
                print(f"  Median error: {stats['median_error']:.2f} km")
                print(f"  Success rate: {stats['success_rate']:.1f}%")
            else:
                print(f"\n{feature.title()} Removed:")
                print(f"  Average error increase: {stats['avg_error_increase']:.2f} km")
                print(f"  Median error increase: {stats['median_error_increase']:.2f} km")
                print(f"  Max error increase: {stats['max_error_increase']:.2f} km")
                print(f"  Impact severity: {stats['impact_severity']}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate geolocation feature importance using GPT vision models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate impact of removing people and vehicles
  %(prog)s -f "person,vehicle" ./sample/ results.db
  
  # Full evaluation with all features
  %(prog)s -f "all" ./dataset/ comprehensive_results.db
  
  # Test run with limited images
  %(prog)s -f "person" --max-images 10 ./sample/ test_results.db
        """
    )
    
    parser.add_argument('input_directory', 
                       help='Directory containing images, masks, and results')
    parser.add_argument('results_database',
                       help='SQLite database file to store results')
    
    parser.add_argument('-f', '--features', default='person,vehicle',
                       help='Comma-separated features to test (default: person,vehicle)')
    parser.add_argument('-m', '--model', default='gpt-4-vision-preview',
                       help='OpenAI model to use (default: gpt-4-vision-preview)')
    parser.add_argument('-c', '--concurrent', type=int, default=5,
                       help='Number of concurrent API calls (default: 5)')
    parser.add_argument('--original-only', action='store_true',
                       help='Only evaluate original images (skip feature removal)')
    parser.add_argument('--cache-responses', action='store_true',
                       help='Cache GPT responses to avoid redundant API calls')
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to process (for testing)')
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Validate input directory
    if not os.path.isdir(args.input_directory):
        printError(f"Input directory does not exist: {args.input_directory}")
    
    # Create results database directory if needed
    db_dir = os.path.dirname(args.results_database)
    if db_dir:
        ensurePath(db_dir)
    
    # Create and run evaluator
    evaluator = GeoLocationFeatureEvaluator(args)
    await evaluator.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
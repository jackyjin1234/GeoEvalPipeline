#!/usr/bin/env python3
"""
Dataset Functionality Tests

Tests for dataset discovery, validation, selection algorithms,
and GPS coordinate handling.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch

from core.pipeline_dataset import PipelineDataset, ImageItem


class TestPipelineDataset(unittest.TestCase):
    """Test dataset discovery and management functionality"""
    
    def setUp(self):
        """Set up test environment with mock dataset"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="dataset_test_"))
        self.addCleanup(self._cleanup_test_dir)
        
        # Create mock Cityscapes structure
        self._create_mock_cityscapes_structure()
        
        # Create dataset manager
        self.dataset = PipelineDataset(str(self.test_dir))
    
    def _cleanup_test_dir(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_mock_cityscapes_structure(self):
        """Create mock Cityscapes dataset structure"""
        # Create directory structure
        cities = ['aachen', 'bremen', 'cologne']
        
        for city in cities:
            # Create directories
            images_dir = self.test_dir / 'leftImg8bit' / 'train' / city
            annotations_dir = self.test_dir / 'gtFine' / 'train' / city
            gps_dir = self.test_dir / 'vehicle' / 'train' / city
            
            for dir_path in [images_dir, annotations_dir, gps_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create mock files for each city
            num_files = 5 if city == 'aachen' else 3
            
            for i in range(num_files):
                base_name = f"{city}_{i:06d}_{19:06d}"
                
                # Create image file
                image_file = images_dir / f"{base_name}_leftImg8bit.png"
                image_file.touch()
                
                # Create annotation file
                annotation_file = annotations_dir / f"{base_name}_gtFine_polygons.json"
                self._create_mock_annotation_file(annotation_file)
                
                # Create GPS file
                gps_file = gps_dir / f"{base_name}_vehicle.json"
                self._create_mock_gps_file(gps_file, 50.0 + i, 6.0 + i)
        
        # Create some incomplete triplets for testing validation
        incomplete_city = 'dusseldorf'
        images_dir = self.test_dir / 'leftImg8bit' / 'train' / incomplete_city
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Only create image file, missing annotation and GPS
        image_file = images_dir / f"{incomplete_city}_000000_000019_leftImg8bit.png"
        image_file.touch()
    
    def _create_mock_annotation_file(self, path: Path):
        """Create mock annotation JSON file"""
        annotation_data = {
            "imgHeight": 1024,
            "imgWidth": 2048,
            "objects": [
                {
                    "label": "person",
                    "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                    "deleted": False
                },
                {
                    "label": "car",
                    "polygon": [[300, 300], [500, 300], [500, 400], [300, 400]],
                    "deleted": False
                }
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(annotation_data, f)
    
    def _create_mock_gps_file(self, path: Path, lat: float, lon: float):
        """Create mock GPS JSON file"""
        gps_data = {
            "gpsLatitude": lat,
            "gpsLongitude": lon,
            "gpsHeading": 90.0,
            "outsideTemperature": 20.0,
            "speed": 30.0,
            "yawRate": 0.1
        }
        
        with open(path, 'w') as f:
            json.dump(gps_data, f)
    
    def test_file_discovery(self):
        """Test finding matching image/annotation/GPS files"""
        print("\n=== Testing File Discovery ===")
        
        # Discover all triplets
        items = self.dataset.discover_all_triplets()
        
        # Should find items from complete cities only
        self.assertGreater(len(items), 0, "Should discover some items")
        
        # Check that all discovered items are valid
        for item in items:
            self.assertIsInstance(item, ImageItem)
            self.assertTrue(item.image_path.exists(), f"Image should exist: {item.image_path}")
            self.assertTrue(item.annotation_path.exists(), f"Annotation should exist: {item.annotation_path}")
            self.assertTrue(item.gps_path.exists(), f"GPS should exist: {item.gps_path}")
        
        # Verify expected counts
        expected_total = 5 + 3 + 3  # aachen + bremen + cologne
        self.assertEqual(len(items), expected_total, f"Should find {expected_total} complete triplets")
    
    def test_triplet_validation(self):
        """Test validation of complete file triplets"""
        print("\n=== Testing Triplet Validation ===")
        
        items = self.dataset.discover_all_triplets()
        
        # Test individual item validation
        for item in items:
            self.assertTrue(item.is_valid(), f"Item should be valid: {item.image_id}")
        
        # Test validation report
        validation = self.dataset.validate_selection(items)
        self.assertTrue(validation['all_files_exist'], "All files should exist")
        self.assertTrue(validation['all_have_gps'], "All items should have GPS")
        self.assertEqual(len(validation['cities_represented']), 3, "Should represent 3 cities")
    
    def test_gps_extraction(self):
        """Test GPS coordinate extraction from vehicle JSON"""
        print("\n=== Testing GPS Extraction ===")
        
        items = self.dataset.discover_all_triplets()
        
        for item in items:
            # Should have GPS coordinates
            self.assertIsNotNone(item.ground_truth_coords, f"Should have GPS: {item.image_id}")
            
            lat, lon = item.ground_truth_coords
            
            # Validate coordinate ranges
            self.assertTrue(-90 <= lat <= 90, f"Invalid latitude: {lat}")
            self.assertTrue(-180 <= lon <= 180, f"Invalid longitude: {lon}")
            
            # Check that coordinates make sense for test data
            self.assertTrue(45 <= lat <= 55, f"Latitude should be in test range: {lat}")
            self.assertTrue(0 <= lon <= 12, f"Longitude should be in test range: {lon}")
    
    def test_selection_algorithms(self):
        """Test different image selection strategies"""
        print("\n=== Testing Selection Algorithms ===")
        
        # Test first_n selection
        selected_first = self.dataset.select_images_per_city(
            cities=['aachen'],
            images_per_city=3,
            selection_method='first_n'
        )
        self.assertEqual(len(selected_first), 3, "Should select 3 items with first_n")
        
        # Test systematic selection
        selected_systematic = self.dataset.select_images_per_city(
            cities=['aachen'],
            images_per_city=3,
            selection_method='systematic'
        )
        self.assertEqual(len(selected_systematic), 3, "Should select 3 items with systematic")
        
        # Test evenly_spaced selection
        selected_spaced = self.dataset.select_images_per_city(
            cities=['aachen'],
            images_per_city=3,
            selection_method='evenly_spaced'
        )
        self.assertEqual(len(selected_spaced), 3, "Should select 3 items with evenly_spaced")
        
        # Verify different methods can produce different results
        # (with enough items, systematic and evenly_spaced should differ from first_n)
        first_ids = {item.image_id for item in selected_first}
        systematic_ids = {item.image_id for item in selected_systematic}
        
        # With 5 items selecting 3, first_n takes [0,1,2], systematic takes [0,1,3]
        self.assertNotEqual(first_ids, systematic_ids, "Different methods should produce different selections")
    
    def test_data_structure_consistency(self):
        """Test ImageItem data structure integrity"""
        print("\n=== Testing Data Structure Consistency ===")
        
        items = self.dataset.discover_all_triplets()
        
        for item in items:
            # Test basic attributes
            self.assertIsInstance(item.image_id, str)
            self.assertIsInstance(item.city, str)
            self.assertIsInstance(item.image_path, Path)
            self.assertIsInstance(item.annotation_path, Path)
            self.assertIsInstance(item.gps_path, Path)
            
            # Test GPS coordinates
            if item.ground_truth_coords:
                self.assertIsInstance(item.ground_truth_coords, tuple)
                self.assertEqual(len(item.ground_truth_coords), 2)
                lat, lon = item.ground_truth_coords
                self.assertIsInstance(lat, float)
                self.assertIsInstance(lon, float)
            
            # Test collections are initialized
            self.assertIsInstance(item.masks, dict)
            self.assertIsInstance(item.processed_images, dict)
            self.assertIsInstance(item.split_images, dict)
            self.assertIsInstance(item.clipaway_results, dict)
            self.assertIsInstance(item.evaluation_results, dict)
            
            # Test base filename generation
            base_name = item.get_base_filename()
            self.assertIsInstance(base_name, str)
            self.assertNotIn('_leftImg8bit', base_name)
    
    def test_coordinate_ranges_calculation(self):
        """Test geographic coordinate range calculations"""
        print("\n=== Testing Coordinate Ranges ===")
        
        items = self.dataset.discover_all_triplets()
        stats = self.dataset.get_statistics()
        
        # Should have coordinate range information
        self.assertIn('coordinate_ranges', stats)
        coord_ranges = stats['coordinate_ranges']
        
        if coord_ranges['lat_range']:
            lat_min, lat_max = coord_ranges['lat_range']
            lon_min, lon_max = coord_ranges['lon_range']
            
            # Validate ranges
            self.assertLessEqual(lat_min, lat_max)
            self.assertLessEqual(lon_min, lon_max)
            
            # Check center calculation
            center_lat, center_lon = coord_ranges['center']
            self.assertTrue(lat_min <= center_lat <= lat_max)
            self.assertTrue(lon_min <= center_lon <= lon_max)
    
    def test_missing_files_handling(self):
        """Test handling of missing or incomplete file triplets"""
        print("\n=== Testing Missing Files Handling ===")
        
        # Create item with missing files
        missing_item = ImageItem(
            image_id='missing_test',
            city='test_city',
            image_path=Path('/nonexistent/image.png'),
            annotation_path=Path('/nonexistent/annotation.json'),
            gps_path=Path('/nonexistent/gps.json')
        )
        
        # Should not be valid
        self.assertFalse(missing_item.is_valid(), "Item with missing files should not be valid")
        self.assertIsNone(missing_item.ground_truth_coords, "Should not have GPS coordinates")
    
    def test_selection_edge_cases(self):
        """Test selection with edge cases"""
        print("\n=== Testing Selection Edge Cases ===")
        
        # Request more images than available
        selected = self.dataset.select_images_per_city(
            cities=['bremen'],  # Only has 3 items
            images_per_city=5,  # Request 5
            selection_method='first_n'
        )
        self.assertEqual(len(selected), 3, "Should return all available items when requesting more than available")
        
        # Request zero images
        selected_zero = self.dataset.select_images_per_city(
            cities=['bremen'],
            images_per_city=0,
            selection_method='first_n'
        )
        self.assertEqual(len(selected_zero), 0, "Should return empty list when requesting zero items")
        
        # Non-existent city
        selected_missing = self.dataset.select_images_per_city(
            cities=['nonexistent_city'],
            images_per_city=3,
            selection_method='first_n'
        )
        self.assertEqual(len(selected_missing), 0, "Should return empty list for non-existent city")
    
    def test_statistics_generation(self):
        """Test dataset statistics generation"""
        print("\n=== Testing Statistics Generation ===")
        
        stats = self.dataset.get_statistics()
        
        # Check required fields
        required_fields = ['total_items', 'cities', 'items_per_city', 'valid_gps_count', 'coordinate_ranges']
        for field in required_fields:
            self.assertIn(field, stats, f"Statistics should include {field}")
        
        # Validate statistics content
        self.assertGreater(stats['total_items'], 0)
        print(stats['cities'])
        self.assertEqual(len(stats['cities']), 3)  # aachen, bremen, cologne
        self.assertEqual(stats['valid_gps_count'], stats['total_items'])  # All should have GPS
        
        # Check per-city statistics
        self.assertEqual(stats['items_per_city']['aachen'], 5)
        self.assertEqual(stats['items_per_city']['bremen'], 3)
        self.assertEqual(stats['items_per_city']['cologne'], 3)
    
    def test_geographic_span_calculation(self):
        """Test geographic span calculations"""
        print("\n=== Testing Geographic Span ===")
        
        items = self.dataset.discover_all_triplets()
        selected = items[:5]  # Take first 5 items
        
        validation = self.dataset.validate_selection(selected)
        
        if validation.get('coordinate_coverage'):
            coverage = validation['coordinate_coverage']
            
            # Should have coordinate ranges
            self.assertIn('lat_range', coverage)
            self.assertIn('lon_range', coverage)
            self.assertIn('span_km', coverage)
            
            # Span should be reasonable
            span_km = coverage['span_km']
            self.assertGreaterEqual(span_km, 0)
            
            # For our test data (coordinates 0-11 degrees apart), span should be substantial
            self.assertGreater(span_km, 100)  # Should span more than 100km


class TestImageItem(unittest.TestCase):
    """Test ImageItem class functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="imageitem_test_"))
        self.addCleanup(self._cleanup_test_dir)
    
    def _cleanup_test_dir(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_imageitem_creation(self):
        """Test ImageItem creation and initialization"""
        print("\n=== Testing ImageItem Creation ===")
        
        # Create test files
        image_path = self.test_dir / 'test_image.png'
        annotation_path = self.test_dir / 'test_annotation.json'
        gps_path = self.test_dir / 'test_gps.json'
        
        image_path.touch()
        
        # Create annotation file
        annotation_data = {"imgHeight": 1024, "imgWidth": 2048, "objects": []}
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f)
        
        # Create GPS file
        gps_data = {"gpsLatitude": 50.0, "gpsLongitude": 6.0}
        with open(gps_path, 'w') as f:
            json.dump(gps_data, f)
        
        # Create ImageItem
        item = ImageItem(
            image_id='test_item',
            city='test_city',
            image_path=image_path,
            annotation_path=annotation_path,
            gps_path=gps_path
        )
        
        # Test initialization
        self.assertEqual(item.image_id, 'test_item')
        self.assertEqual(item.city, 'test_city')
        self.assertEqual(item.image_path, image_path)
        self.assertEqual(item.annotation_path, annotation_path)
        self.assertEqual(item.gps_path, gps_path)
        
        # Should extract GPS coordinates
        self.assertIsNotNone(item.ground_truth_coords)
        self.assertEqual(item.ground_truth_coords, (50.0, 6.0))
        
        # Should be valid
        self.assertTrue(item.is_valid())
    
    def test_gps_extraction_edge_cases(self):
        """Test GPS coordinate extraction with various edge cases"""
        print("\n=== Testing GPS Extraction Edge Cases ===")
        
        # Test with missing GPS file
        item_no_gps = ImageItem(
            image_id='no_gps',
            city='test',
            image_path=Path('/nonexistent/image.png'),
            annotation_path=Path('/nonexistent/annotation.json'),
            gps_path=Path('/nonexistent/gps.json')
        )
        self.assertIsNone(item_no_gps.ground_truth_coords)
        
        # Test with invalid GPS data
        invalid_gps_path = self.test_dir / 'invalid_gps.json'
        with open(invalid_gps_path, 'w') as f:
            json.dump({"invalid": "data"}, f)
        
        item_invalid_gps = ImageItem(
            image_id='invalid_gps',
            city='test',
            image_path=Path('/nonexistent/image.png'),
            annotation_path=Path('/nonexistent/annotation.json'),
            gps_path=invalid_gps_path
        )
        self.assertIsNone(item_invalid_gps.ground_truth_coords)
        
        # Test with partial GPS data (missing longitude)
        partial_gps_path = self.test_dir / 'partial_gps.json'
        with open(partial_gps_path, 'w') as f:
            json.dump({"gpsLatitude": 50.0}, f)
        
        item_partial_gps = ImageItem(
            image_id='partial_gps',
            city='test',
            image_path=Path('/nonexistent/image.png'),
            annotation_path=Path('/nonexistent/annotation.json'),
            gps_path=partial_gps_path
        )
        self.assertIsNone(item_partial_gps.ground_truth_coords)
    
    def test_base_filename_generation(self):
        """Test base filename generation"""
        print("\n=== Testing Base Filename Generation ===")
        
        # Test with typical Cityscapes filename
        image_path = Path('/path/to/aachen_000001_000019_leftImg8bit.png')
        item = ImageItem(
            image_id='test',
            city='aachen',
            image_path=image_path,
            annotation_path=Path('/test/annotation.json'),
            gps_path=Path('/test/gps.json')
        )
        
        base_name = item.get_base_filename()
        self.assertEqual(base_name, 'aachen_000001_000019')
        self.assertNotIn('_leftImg8bit', base_name)
    
    def test_artifact_management(self):
        """Test management of processing artifacts"""
        print("\n=== Testing Artifact Management ===")
        
        item = ImageItem(
            image_id='test_artifacts',
            city='test',
            image_path=Path('/test/image.png'),
            annotation_path=Path('/test/annotation.json'),
            gps_path=Path('/test/gps.json')
        )
        
        # Test initial state
        self.assertEqual(len(item.masks), 0)
        self.assertEqual(len(item.processed_images), 0)
        self.assertEqual(len(item.clipaway_results), 0)
        self.assertEqual(len(item.evaluation_results), 0)
        
        # Test adding artifacts
        item.masks['human'] = Path('/masks/human_mask.png')
        item.processed_images['original'] = {'left': Path('/left.png'), 'right': Path('/right.png')}
        item.clipaway_results['human'] = Path('/results/human_removed.jpg')
        item.evaluation_results['human'] = {'error': 5.2}
        
        # Verify artifacts were added
        self.assertEqual(len(item.masks), 1)
        self.assertIn('human', item.masks)
        self.assertIn('original', item.processed_images)
        self.assertIn('human', item.clipaway_results)
        self.assertIn('human', item.evaluation_results)


def run_dataset_tests():
    """Run all dataset tests"""
    print("Starting Dataset Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestPipelineDataset))
    suite.addTest(unittest.makeSuite(TestImageItem))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    
    success = run_dataset_tests()
    sys.exit(0 if success else 1)
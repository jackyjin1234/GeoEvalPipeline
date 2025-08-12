#!/usr/bin/env python3
"""
Adapter Integration Tests

Tests for all pipeline adapters that integrate existing Cityscapes tools
with the optimized pipeline architecture.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.pipeline_dataset import ImageItem
from adapters.mask_generator import MaskGeneratorAdapter
from adapters.image_processor import ImageProcessorAdapter
from adapters.clipaway_adapter import CLIPAwayAdapter
from adapters.evaluator_adapter import EvaluatorAdapter


class TestMaskGeneratorAdapter(unittest.TestCase):
    """Test binary mask generation adapter"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="mask_test_"))
        self.addCleanup(self._cleanup_test_dir)
        
        # Create test items
        self.test_items = self._create_test_items()
    
    def _cleanup_test_dir(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_items(self):
        """Create test ImageItem objects"""
        items = []
        
        for i in range(2):
            # Create annotation file
            annotation_path = self.test_dir / f'test_{i}_annotation.json'
            self._create_mock_annotation(annotation_path)
            
            item = ImageItem(
                image_id=f'test_{i}',
                city='test_city',
                image_path=self.test_dir / f'test_{i}_image.png',
                annotation_path=annotation_path,
                gps_path=self.test_dir / f'test_{i}_gps.json'
            )
            items.append(item)
        
        return items
    
    def _create_mock_annotation(self, path: Path):
        """Create mock annotation file"""
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
    
    @patch('pipeline.adapters.mask_generator.getSelectedLabels')
    @patch('pipeline.adapters.mask_generator.createBinaryMask')
    @patch('pipeline.adapters.mask_generator.Annotation')
    def test_mask_generator_integration(self, mock_annotation, mock_create_mask, mock_get_labels):
        """Test mask generator integration with mocked cityscapes functions"""
        print("\n=== Testing Mask Generator Integration ===")
        
        # Setup mocks
        mock_label = Mock()
        mock_label.name = 'person'
        mock_get_labels.return_value = [mock_label]
        
        mock_mask = Mock()
        mock_create_mask.return_value = {'combined': mock_mask}
        
        mock_annotation_instance = Mock()
        mock_annotation_instance.imgWidth = 2048
        mock_annotation_instance.imgHeight = 1024
        mock_annotation.return_value = mock_annotation_instance
        
        # Create adapter
        adapter = MaskGeneratorAdapter(
            target_features=['human', 'vehicle'],
            combine_masks=False
        )
        
        # Test batch processing
        output_dir = self.test_dir / 'masks'
        results = adapter.generate_masks_batch(self.test_items, output_dir)
        
        # Verify results structure
        self.assertIn('human', results)
        self.assertIn('vehicle', results)
        self.assertEqual(len(results['human']), len(self.test_items))
        self.assertEqual(len(results['vehicle']), len(self.test_items))
    
    def test_feature_validation(self):
        """Test feature validation during adapter initialization"""
        print("\n=== Testing Feature Validation ===")
        
        # Test with valid features (mocked)
        with patch('adapters.mask_generator.getSelectedLabels') as mock_get_labels:
            mock_label = Mock()
            mock_get_labels.return_value = [mock_label]
            
            adapter = MaskGeneratorAdapter(['human'])
            self.assertEqual(adapter.target_features, ['human'])
        
        # Test with invalid features
        with patch('adapters.mask_generator.getSelectedLabels') as mock_get_labels:
            mock_get_labels.return_value = []
            
            with self.assertRaises(ValueError):
                MaskGeneratorAdapter(['invalid_feature'])
    
    def test_mask_validation(self):
        """Test mask validation functionality"""
        print("\n=== Testing Mask Validation ===")
        
        # Create adapter
        adapter = MaskGeneratorAdapter(['human'])
        
        # Create test items with mock mask paths
        for item in self.test_items:
            mask_path = self.test_dir / f'{item.image_id}_mask.png'
            
            # Create mock mask image
            try:
                from PIL import Image
                import numpy as np
                mask_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
                mask_img = Image.fromarray(mask_array, mode='L')
                mask_img.save(mask_path)
            except ImportError:
                # Fallback - create empty file
                mask_path.touch()
            
            item.masks['human'] = mask_path
        
        # Test validation
        validation = adapter.validate_masks(self.test_items)
        
        self.assertIn('total_items', validation)
        self.assertIn('items_with_all_masks', validation)
        self.assertIn('success_rate', validation)
        self.assertEqual(validation['total_items'], len(self.test_items))


class TestImageProcessorAdapter(unittest.TestCase):
    """Test image processing adapter"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="image_proc_test_"))
        self.addCleanup(self._cleanup_test_dir)
        
        # Create adapter
        self.adapter = ImageProcessorAdapter(target_size=256)
        
        # Create test items with mock images
        self.test_items = self._create_test_items_with_images()
    
    def _cleanup_test_dir(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_items_with_images(self):
        """Create test items with mock images"""
        items = []
        
        for i in range(2):
            # Create mock image
            image_path = self.test_dir / f'test_{i}_image.png'
            self._create_mock_image(image_path, (2048, 1024))
            
            item = ImageItem(
                image_id=f'test_{i}',
                city='test_city',
                image_path=image_path,
                annotation_path=self.test_dir / f'test_{i}_annotation.json',
                gps_path=self.test_dir / f'test_{i}_gps.json'
            )
            items.append(item)
        
        return items
    
    def _create_mock_image(self, path: Path, size: tuple = (2048, 1024)):
        """Create mock image file"""
        try:
            from PIL import Image
            import numpy as np
            
            img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(path)
        except ImportError:
            # Fallback - create empty file
            path.touch()
    
    @patch('pipeline.adapters.image_processor.downscale')
    def test_image_processor_integration(self, mock_downscale):
        """Test image processor integration with existing downscale function"""
        print("\n=== Testing Image Processor Integration ===")
        
        # Setup mock
        def mock_downscale_func(input_path, left_path, right_path):
            # Create mock output files
            Path(left_path).touch()
            Path(right_path).touch()
        
        mock_downscale.side_effect = mock_downscale_func
        
        # Test processing
        output_dir = self.test_dir / 'processed'
        results = self.adapter.process_images_batch(
            self.test_items,
            output_dir,
            process_masks=False
        )
        
        # Verify results
        self.assertIn('original_left', results)
        self.assertIn('original_right', results)
        self.assertEqual(len(results['original_left']), len(self.test_items))
        self.assertEqual(len(results['original_right']), len(self.test_items))
        
        # Verify downscale was called
        self.assertEqual(mock_downscale.call_count, len(self.test_items))
    
    def test_size_validation(self):
        """Test image size validation"""
        print("\n=== Testing Size Validation ===")
        
        # Create test image with known size
        test_image_path = self.test_dir / 'size_test.png'
        self._create_mock_image(test_image_path, (256, 256))
        
        # Test validation
        is_valid = self.adapter._validate_image_size(test_image_path)
        self.assertTrue(is_valid, "Should validate correct size")
        
        # Test with wrong size
        wrong_size_path = self.test_dir / 'wrong_size.png'
        self._create_mock_image(wrong_size_path, (512, 512))
        
        is_valid_wrong = self.adapter._validate_image_size(wrong_size_path)
        self.assertFalse(is_valid_wrong, "Should reject wrong size")
    
    def test_mask_processing(self):
        """Test mask processing functionality"""
        print("\n=== Testing Mask Processing ===")
        
        # Add mock masks to test items
        for item in self.test_items:
            mask_path = self.test_dir / f'{item.image_id}_mask.png'
            self._create_mock_image(mask_path, (1024, 1024))  # Square mask
            item.masks['human'] = mask_path
        
        # Test processing with masks
        output_dir = self.test_dir / 'processed_masks'
        
        with patch('adapters.image_processor.downscale') as mock_downscale:
            def mock_downscale_func(input_path, left_path, right_path):
                Path(left_path).touch()
                Path(right_path).touch()
            mock_downscale.side_effect = mock_downscale_func
            
            results = self.adapter.process_images_batch(
                self.test_items,
                output_dir,
                process_masks=True
            )
            
            # Should have mask results
            self.assertIn('mask_results', results)
            if results['mask_results']:
                self.assertIn('human', results['mask_results'])


class TestCLIPAwayAdapter(unittest.TestCase):
    """Test CLIPAway integration adapter"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="clipaway_test_"))
        self.addCleanup(self._cleanup_test_dir)
    
    def _cleanup_test_dir(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @patch('pipeline.adapters.clipaway_adapter.CLIPAWAY_AVAILABLE', False)
    def test_clipaway_unavailable_handling(self):
        """Test handling when CLIPAway dependencies are unavailable"""
        print("\n=== Testing CLIPAway Unavailable Handling ===")
        
        with self.assertRaises(RuntimeError):
            CLIPAwayAdapter()
    
    @patch('pipeline.adapters.clipaway_adapter.CLIPAWAY_AVAILABLE', True)
    @patch('pipeline.adapters.clipaway_adapter.CLIPAway')
    @patch('pipeline.adapters.clipaway_adapter.StableDiffusionInpaintPipeline')
    def test_clipaway_model_loading(self, mock_sd_pipeline, mock_clipaway):
        """Test CLIPAway model loading with mocks"""
        print("\n=== Testing CLIPAway Model Loading ===")
        
        # Create adapter
        adapter = CLIPAwayAdapter(device='cpu')
        
        # Test model info before loading
        info = adapter.get_model_info()
        self.assertFalse(info['models_loaded'])
        
        # Test model loading
        adapter._load_models()
        
        # Verify models were initialized
        mock_sd_pipeline.from_pretrained.assert_called_once()
        mock_clipaway.assert_called_once()
    
    def test_model_cleanup(self):
        """Test model cleanup functionality"""
        print("\n=== Testing Model Cleanup ===")
        
        # Create adapter with mocked availability
        with patch('pipeline.adapters.clipaway_adapter.CLIPAWAY_AVAILABLE', True):
            adapter = CLIPAwayAdapter(device='cpu')
            
            # Test cleanup (should not raise errors even if models not loaded)
            adapter.cleanup_models()
            
            # Verify cleanup completed
            self.assertIsNone(adapter.clipaway_model)
            self.assertIsNone(adapter.sd_pipeline)


class TestEvaluatorAdapter(unittest.TestCase):
    """Test geolocation evaluator adapter"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="evaluator_test_"))
        self.addCleanup(self._cleanup_test_dir)
        
        # Create adapter in mock mode
        self.adapter = EvaluatorAdapter(mock_mode=True)
        
        # Create test items
        self.test_items = self._create_test_items()
    
    def _cleanup_test_dir(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_items(self):
        """Create test items with mock data"""
        items = []
        
        for i in range(2):
            # Create mock image
            image_path = self.test_dir / f'test_{i}_image.jpg'
            image_path.touch()
            
            # Create mock GPS data
            gps_path = self.test_dir / f'test_{i}_gps.json'
            gps_data = {"gpsLatitude": 50.0 + i, "gpsLongitude": 6.0 + i}
            with open(gps_path, 'w') as f:
                json.dump(gps_data, f)
            
            item = ImageItem(
                image_id=f'test_{i}',
                city='test_city',
                image_path=image_path,
                annotation_path=self.test_dir / f'test_{i}_annotation.json',
                gps_path=gps_path
            )
            
            # Add mock CLIPAway results
            clipaway_result = self.test_dir / f'test_{i}_human_removed.jpg'
            clipaway_result.touch()
            item.clipaway_results['human'] = clipaway_result
            
            items.append(item)
        
        return items
    
    async def test_mock_evaluation_api(self):
        """Test mock evaluation API functionality"""
        print("\n=== Testing Mock Evaluation API ===")
        
        # Test single prediction
        prediction = await self.adapter.api.get_geolocation_prediction('test_image.jpg')
        
        self.assertIsNotNone(prediction)
        self.assertIn('coordinates', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('reasoning', prediction)
        
        # Validate coordinate format
        lat, lon = prediction['coordinates']
        self.assertIsInstance(lat, float)
        self.assertIsInstance(lon, float)
        self.assertTrue(-90 <= lat <= 90)
        self.assertTrue(-180 <= lon <= 180)
    
    async def test_batch_evaluation(self):
        """Test batch evaluation functionality"""
        print("\n=== Testing Batch Evaluation ===")
        
        # Run batch evaluation
        output_dir = self.test_dir / 'evaluation'
        results = await self.adapter.evaluate_items_batch(
            self.test_items,
            ['human'],
            output_dir
        )
        
        # Verify results structure
        self.assertIn('total_items', results)
        self.assertIn('completed_evaluations', results)
        self.assertIn('results_by_item', results)
        self.assertIn('summary_statistics', results)
        
        # Should have evaluated all items
        self.assertEqual(results['total_items'], len(self.test_items))
        self.assertGreater(results['completed_evaluations'], 0)
    
    def test_distance_calculation(self):
        """Test distance calculation functionality"""
        print("\n=== Testing Distance Calculation ===")
        
        # Test with known coordinates
        coord1 = (50.0, 6.0)
        coord2 = (51.0, 7.0)
        
        distance = self.adapter._calculate_distance(coord1, coord2)
        
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
        
        # Distance should be reasonable (roughly 111km per degree)
        self.assertTrue(100 < distance < 200)  # Approximately 156km
    
    def test_impact_categorization(self):
        """Test impact severity categorization"""
        print("\n=== Testing Impact Categorization ===")
        
        # Test different impact levels
        test_cases = [
            (0.5, "Minimal"),
            (3.0, "Low"),
            (15.0, "Moderate"),
            (50.0, "High"),
            (150.0, "Critical")
        ]
        
        for error_increase, expected_category in test_cases:
            category = self.adapter._categorize_impact(error_increase)
            self.assertEqual(category, expected_category, 
                           f"Error increase {error_increase} should be {expected_category}")
    
    def test_evaluation_setup_validation(self):
        """Test evaluation setup validation"""
        print("\n=== Testing Evaluation Setup Validation ===")
        
        validation = self.adapter.validate_evaluation_setup()
        
        self.assertIn('valid', validation)
        self.assertIn('api_available', validation)
        self.assertIn('database_available', validation)
        
        # Mock mode should be indicated
        self.assertFalse(validation['api_available'])
        self.assertIn("Using mock API", str(validation['warnings']))


class TestAdapterIntegration(unittest.TestCase):
    """Test adapter integration and coordination"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="adapter_integration_"))
        self.addCleanup(self._cleanup_test_dir)
    
    def _cleanup_test_dir(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_adapter_data_flow(self):
        """Test data flow between adapters"""
        print("\n=== Testing Adapter Data Flow ===")
        
        # Create test item
        item = ImageItem(
            image_id='integration_test',
            city='test_city',
            image_path=self.test_dir / 'test_image.png',
            annotation_path=self.test_dir / 'test_annotation.json',
            gps_path=self.test_dir / 'test_gps.json'
        )
        
        # Simulate adapter outputs being added to item
        
        # 1. Mask generator output
        mask_path = self.test_dir / 'human_mask.png'
        mask_path.touch()
        item.masks['human'] = mask_path
        
        # 2. Image processor output
        left_path = self.test_dir / 'left.png'
        right_path = self.test_dir / 'right.png'
        left_path.touch()
        right_path.touch()
        item.processed_images['original'] = {'left': left_path, 'right': right_path}
        item.processed_images['human'] = {'left': left_path, 'right': right_path}
        
        # 3. CLIPAway output
        clipaway_path = self.test_dir / 'human_removed.jpg'
        clipaway_path.touch()
        item.clipaway_results['human'] = clipaway_path
        
        # 4. Evaluation output
        item.evaluation_results['human'] = {'error_km': 5.2, 'confidence': 'High'}
        
        # Verify complete data flow
        self.assertIn('human', item.masks)
        self.assertIn('original', item.processed_images)
        self.assertIn('human', item.processed_images)
        self.assertIn('human', item.clipaway_results)
        self.assertIn('human', item.evaluation_results)
        
        # Verify all paths exist
        self.assertTrue(item.masks['human'].exists())
        self.assertTrue(item.processed_images['original']['left'].exists())
        self.assertTrue(item.clipaway_results['human'].exists())
    
    def test_error_propagation(self):
        """Test error handling and propagation between adapters"""
        print("\n=== Testing Error Propagation ===")
        
        # Test with missing input files
        item = ImageItem(
            image_id='error_test',
            city='test_city',
            image_path=Path('/nonexistent/image.png'),
            annotation_path=Path('/nonexistent/annotation.json'),
            gps_path=Path('/nonexistent/gps.json')
        )
        
        # Should not be valid
        self.assertFalse(item.is_valid())
        
        # Adapters should handle missing files gracefully
        # (This would be tested in actual adapter methods with proper error handling)


def run_adapter_tests():
    """Run all adapter tests"""
    print("Starting Adapter Integration Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestMaskGeneratorAdapter))
    suite.addTest(unittest.makeSuite(TestImageProcessorAdapter))
    suite.addTest(unittest.makeSuite(TestCLIPAwayAdapter))
    suite.addTest(unittest.makeSuite(TestEvaluatorAdapter))
    suite.addTest(unittest.makeSuite(TestAdapterIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ADAPTER TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    
    success = run_adapter_tests()
    sys.exit(0 if success else 1)
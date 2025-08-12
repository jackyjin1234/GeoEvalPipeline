#!/usr/bin/env python3
"""
Comprehensive Pipeline Integration Tests

This test suite provides end-to-end validation of the visual cue evaluation
pipeline with systematic testing of all components and phases.
"""

import asyncio
import logging
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
from core.pipeline_config import PipelineConfig
from core.pipeline_dataset import PipelineDataset, ImageItem
from core.visual_cue_pipeline import VisualCuePipeline
from core.pipeline_state import PipelineState, PipelinePhase, PhaseStatus
from core.pipeline_utils import setup_logging

# Suppress logging for tests
logging.getLogger().setLevel(logging.CRITICAL)


class TestVisualCuePipeline(unittest.IsolatedAsyncioTestCase):
    """Comprehensive integration tests for the visual cue pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
        self.addCleanup(self._cleanup_test_dir)
        
        # Create test configuration
        self.config = self._create_test_config()
        
        # Create mock dataset structure
        self._create_mock_dataset()
        
        logging.basicConfig(level=logging.CRITICAL)
    
    def _cleanup_test_dir(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_config(self) -> PipelineConfig:
        """Create test configuration"""
        config_dict = {
            'dataset': {
                'cityscapes_root': str(self.test_dir / 'cityscapes'),
                'images_per_city': 2,
                'cities': ['test_city'],
                'selection_method': 'first_n'
            },
            'features': {
                'target_features': ['human', 'vehicle'],
                'combine_masks': False,
                'mask_suffix': 'mask'
            },
            'processing': {
                'image_size': 256,
                'batch_size': 2
            },
            'clipaway': {
                'device': 'cpu',
                'strength': 1.0,
                'scale': 1,
                'seed': 42
            },
            'evaluation': {
                'model': 'mock',
                'concurrent_requests': 1,
                'cache_responses': False,
                'use_ground_truth': True,
                'max_images': 2
            },
            'output': {
                'base_directory': str(self.test_dir / 'output'),
                'cleanup_intermediate': False,
                'save_debug_info': True
            },
            'logging': {
                'level': 'CRITICAL',
                'console': False
            },
            'performance': {
                'max_workers': 1
            },
            'resume': {
                'enabled': False,
                'state_file': str(self.test_dir / 'test_state.json')
            }
        }
        
        # Save config to file
        config_file = self.test_dir / 'test_config.yaml'
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        return PipelineConfig(str(config_file))
    
    def _create_mock_dataset(self):
        """Create mock dataset structure for testing"""
        cityscapes_root = self.test_dir / 'cityscapes'
        
        # Create directory structure
        images_dir = cityscapes_root / 'leftImg8bit' / 'train' / 'test_city'
        annotations_dir = cityscapes_root / 'gtFine' / 'train' / 'test_city'
        gps_dir = cityscapes_root / 'vehicle' / 'train' / 'test_city'
        
        for dir_path in [images_dir, annotations_dir, gps_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create mock files
        for i in range(3):
            base_name = f"test_city_{i:06d}_{19:06d}"
            
            # Mock image file (create small test image)
            image_file = images_dir / f"{base_name}_leftImg8bit.png"
            self._create_mock_image(image_file, (2048, 1024))
            
            # Mock annotation file
            annotation_file = annotations_dir / f"{base_name}_gtFine_polygons.json"
            self._create_mock_annotation(annotation_file)
            
            # Mock GPS file
            gps_file = gps_dir / f"{base_name}_vehicle.json"
            self._create_mock_gps(gps_file, 50.0 + i, 6.0 + i)
    
    def _create_mock_image(self, path: Path, size: tuple = (2048, 1024)):
        """Create a mock image file"""
        try:
            from PIL import Image
            import numpy as np
            
            # Create test image
            img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(path)
        except ImportError:
            # Fallback - create empty file
            path.touch()
    
    def _create_mock_annotation(self, path: Path):
        """Create a mock annotation file"""
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
        
        import json
        with open(path, 'w') as f:
            json.dump(annotation_data, f)
    
    def _create_mock_gps(self, path: Path, lat: float, lon: float):
        """Create a mock GPS file"""
        gps_data = {
            "gpsLatitude": lat,
            "gpsLongitude": lon,
            "gpsHeading": 90.0,
            "speed": 30.0
        }
        
        import json
        with open(path, 'w') as f:
            json.dump(gps_data, f)
    
    async def test_end_to_end_pipeline(self):
        """Test complete pipeline execution end-to-end"""
        print("\n=== Testing End-to-End Pipeline ===")
        
        # Create pipeline
        pipeline = VisualCuePipeline(self.config)
        
        # Validate setup
        validation = pipeline.validate_setup()
        self.assertTrue(validation['valid'], f"Setup validation failed: {validation['errors']}")
        
        # Mock the components that require external dependencies
        with patch.multiple(
            'pipeline.adapters.mask_generator.MaskGeneratorAdapter',
            generate_masks_batch=Mock(return_value={'human': [Path('mock_mask1.png')], 'vehicle': [Path('mock_mask2.png')]}),
            validate_masks=Mock(return_value={'success_rate': 100.0})
        ), patch.multiple(
            'pipeline.adapters.image_processor.ImageProcessorAdapter',
            process_images_batch=Mock(return_value={'original_left': [Path('left1.png')], 'original_right': [Path('right1.png')]}),
            validate_processed_images=Mock(return_value={'original_success_rate': 100.0})
        ), patch.multiple(
            'pipeline.adapters.clipaway_adapter.CLIPAwayAdapter',
            process_items_batch=Mock(return_value={'human': [Path('removed1.jpg')], 'vehicle': [Path('removed2.jpg')]}),
            validate_results=Mock(return_value={'success_rate': 100.0}),
            cleanup_models=Mock()
        ), patch.multiple(
            'pipeline.adapters.evaluator_adapter.EvaluatorAdapter',
            evaluate_items_batch=Mock(return_value={
                'completed_evaluations': 6,
                'summary_statistics': {'test': 'data'}
            })
        ):
            # Run pipeline
            results = await pipeline.run_pipeline(resume=False)
            
            # Verify results
            self.assertTrue(results['success'])
            self.assertEqual(results['total_items_processed'], 2)
            self.assertGreater(results['success_rate'], 0)
            self.assertIn('execution_time', results)
    
    def test_dataset_discovery(self):
        """Test dataset discovery and validation functionality"""
        print("\n=== Testing Dataset Discovery ===")
        
        # Create dataset manager
        dataset = PipelineDataset(str(self.config.dataset.cityscapes_root))
        
        # Test discovery
        items = dataset.discover_all_triplets()
        self.assertGreater(len(items), 0, "Should discover test items")
        
        # Validate discovered items
        for item in items:
            self.assertIsInstance(item, ImageItem)
            self.assertTrue(item.image_path.exists(), f"Image should exist: {item.image_path}")
            self.assertTrue(item.annotation_path.exists(), f"Annotation should exist: {item.annotation_path}")
            self.assertTrue(item.gps_path.exists(), f"GPS file should exist: {item.gps_path}")
            self.assertIsNotNone(item.ground_truth_coords, "Should have GPS coordinates")
        
        # Test statistics
        stats = dataset.get_statistics()
        self.assertIn('total_items', stats)
        self.assertIn('cities', stats)
        self.assertEqual(len(stats['cities']), 1)
        self.assertEqual(stats['cities'][0], 'test_city')
    
    def test_systematic_selection(self):
        """Test systematic image selection algorithms"""
        print("\n=== Testing Systematic Selection ===")
        
        dataset = PipelineDataset(str(self.config.dataset.cityscapes_root))
        
        # Test different selection methods
        methods = ['first_n', 'systematic', 'evenly_spaced']
        
        for method in methods:
            selected = dataset.select_images_per_city(
                cities=['test_city'],
                images_per_city=2,
                selection_method=method
            )
            
            self.assertEqual(len(selected), 2, f"Should select 2 items with {method}")
            
            # Verify all selected items are valid
            for item in selected:
                self.assertTrue(item.is_valid(), f"Selected item should be valid: {item.image_id}")
    
    def test_phase_execution_order(self):
        """Test correct execution order of pipeline phases"""
        print("\n=== Testing Phase Execution Order ===")
        
        # Create pipeline with mocked components
        pipeline = VisualCuePipeline(self.config)
        
        # Check initial phase states
        for phase in PipelinePhase:
            self.assertEqual(pipeline.state.phases[phase].status, PhaseStatus.PENDING)
        
        # The actual phase execution testing would need to mock components
        # This validates the state management structure
        self.assertEqual(len(pipeline.state.phases), len(PipelinePhase))
    
    def test_resume_functionality(self):
        """Test pipeline resume capability"""
        print("\n=== Testing Resume Functionality ===")
        
        # Create state file
        state_file = self.test_dir / 'resume_test_state.json'
        
        # Create initial state
        state = PipelineState(state_file)
        
        # Simulate some progress
        mock_items = [
            ImageItem('test1', 'city1', Path('test1.png'), Path('test1.json'), Path('test1_gps.json')),
            ImageItem('test2', 'city1', Path('test2.png'), Path('test2.json'), Path('test2_gps.json'))
        ]
        
        state.start_pipeline(mock_items)
        state.start_phase(PipelinePhase.DATASET_DISCOVERY, 2)
        state.mark_item_phase_completed('test1', PipelinePhase.DATASET_DISCOVERY)
        state.complete_phase(PipelinePhase.DATASET_DISCOVERY)
        
        # Create new state manager and verify resume
        state2 = PipelineState(state_file)
        
        self.assertEqual(state2.phases[PipelinePhase.DATASET_DISCOVERY].status, PhaseStatus.COMPLETED)
        self.assertTrue(state2.items['test1'].is_phase_completed(PipelinePhase.DATASET_DISCOVERY))
        
        # Test resume report
        report = state2.get_resume_report()
        self.assertIn('Resume Report', report)
        self.assertIn('test1', report)
    
    def test_error_handling(self):
        """Test graceful handling of various error conditions"""
        print("\n=== Testing Error Handling ===")
        
        # Test invalid configuration
        with self.assertRaises(ValueError):
            invalid_config_dict = {'dataset': {'cityscapes_root': '/nonexistent/path'}}
            config_file = self.test_dir / 'invalid_config.yaml'
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(invalid_config_dict, f)
            PipelineConfig(str(config_file))
        
        # Test missing dataset
        config_dict = self.config.to_dict()
        config_dict['dataset']['cityscapes_root'] = '/nonexistent/path'
        config_file = self.test_dir / 'missing_dataset_config.yaml'
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        with self.assertRaises(ValueError):
            PipelineConfig(str(config_file))
    
    def test_output_validation(self):
        """Test output file structure and content validation"""
        print("\n=== Testing Output Validation ===")
        
        # Create output directories
        self.config.create_output_directories()
        
        # Verify directory structure
        expected_dirs = [
            self.config.get_masks_dir(),
            self.config.get_processed_dir(),
            self.config.get_clipaway_dir(),
            self.config.get_evaluation_dir()
        ]
        
        for dir_path in expected_dirs:
            self.assertTrue(dir_path.exists(), f"Output directory should exist: {dir_path}")
    
    def test_ground_truth_integration(self):
        """Test GPS ground truth coordinate extraction and usage"""
        print("\n=== Testing Ground Truth Integration ===")
        
        dataset = PipelineDataset(str(self.config.dataset.cityscapes_root))
        items = dataset.discover_all_triplets()
        
        # Verify GPS coordinate extraction
        for item in items:
            self.assertIsNotNone(item.ground_truth_coords, f"Should have GPS coordinates: {item.image_id}")
            lat, lon = item.ground_truth_coords
            self.assertIsInstance(lat, float)
            self.assertIsInstance(lon, float)
            self.assertTrue(-90 <= lat <= 90, "Latitude should be valid")
            self.assertTrue(-180 <= lon <= 180, "Longitude should be valid")
    
    def test_configuration_validation(self):
        """Test configuration loading and validation"""
        print("\n=== Testing Configuration Validation ===")
        
        # Test valid configuration
        self.assertIsNotNone(self.config.dataset)
        self.assertIsNotNone(self.config.features)
        self.assertIsNotNone(self.config.processing)
        
        # Test configuration constraints
        self.assertGreater(self.config.dataset.images_per_city, 0)
        self.assertGreater(len(self.config.features.target_features), 0)
        self.assertGreater(self.config.processing.image_size, 0)
    
    def test_state_persistence(self):
        """Test pipeline state persistence and recovery"""
        print("\n=== Testing State Persistence ===")
        
        state_file = self.test_dir / 'persistence_test.json'
        
        # Create and populate state
        state1 = PipelineState(state_file)
        mock_items = [ImageItem('test1', 'city1', Path('test1.png'), Path('test1.json'), Path('test1_gps.json'))]
        
        state1.start_pipeline(mock_items)
        state1.start_phase(PipelinePhase.DATASET_DISCOVERY, 1)
        
        # Verify state file exists
        self.assertTrue(state_file.exists())
        
        # Load state and verify persistence
        state2 = PipelineState(state_file)
        self.assertEqual(len(state2.selected_items), 1)
        self.assertEqual(state2.phases[PipelinePhase.DATASET_DISCOVERY].status, PhaseStatus.IN_PROGRESS)
    
    def test_performance_monitoring(self):
        """Test performance monitoring and resource tracking"""
        print("\n=== Testing Performance Monitoring ===")
        
        from core.pipeline_utils import monitor_system_resources, estimate_processing_time
        
        # Test resource monitoring
        resources = monitor_system_resources()
        self.assertIn('cpu_percent', resources)
        self.assertIn('memory_percent', resources)
        
        # Test time estimation
        estimates = estimate_processing_time(5, 7)
        self.assertIn('total', estimates)
        self.assertGreater(estimates['total'], 0)


class TestPipelineComponents(unittest.TestCase):
    """Test individual pipeline components"""
    
    def setUp(self):
        """Set up component tests"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="component_test_"))
        self.addCleanup(self._cleanup_test_dir)
    
    def _cleanup_test_dir(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_configuration_loading(self):
        """Test configuration system"""
        print("\n=== Testing Configuration Loading ===")
        
        # Create test config
        config_dict = {
            'dataset': {'cityscapes_root': str(self.test_dir), 'images_per_city': 1, 'cities': ['test']},
            'features': {'target_features': ['human']},
            'processing': {'image_size': 512},
            'clipaway': {'device': 'cpu'},
            'evaluation': {'model': 'mock'},
            'output': {'base_directory': str(self.test_dir / 'output')},
            'logging': {'level': 'INFO'},
            'performance': {'max_workers': 1},
            'resume': {'enabled': False, 'state_file': 'test.json'}
        }
        
        config_file = self.test_dir / 'test_config.yaml'
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Test loading
        config = PipelineConfig(str(config_file))
        self.assertEqual(config.dataset.images_per_city, 1)
        self.assertEqual(config.features.target_features, ['human'])
        self.assertEqual(config.processing.image_size, 512)
    
    def test_utils_functions(self):
        """Test utility functions"""
        print("\n=== Testing Utility Functions ===")
        
        from core.pipeline_utils import (
            format_bytes, format_duration, create_progress_bar,
            validate_file_integrity, cleanup_directory
        )
        
        # Test formatting functions
        self.assertEqual(format_bytes(1024), "1.0 KB")
        self.assertEqual(format_bytes(1048576), "1.0 MB")
        
        self.assertIn("min", format_duration(120))
        self.assertIn("s", format_duration(30))
        
        # Test progress bar
        progress = create_progress_bar(50, 100)
        self.assertIn("50.0%", progress)
        self.assertIn("50/100", progress)
        
        # Test file validation
        test_file = self.test_dir / 'test.txt'
        test_file.write_text("test content")
        self.assertTrue(validate_file_integrity(test_file))
        self.assertFalse(validate_file_integrity(self.test_dir / 'nonexistent.txt'))


def run_all_tests():
    """Run all pipeline tests"""
    print("Starting Visual Cue Pipeline Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestVisualCuePipeline))
    suite.addTest(unittest.makeSuite(TestPipelineComponents))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    
    # Setup minimal logging for tests
    setup_logging("CRITICAL", console=True)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
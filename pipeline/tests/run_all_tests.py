#!/usr/bin/env python3
"""
Comprehensive Test Runner

Runs all pipeline tests with detailed reporting and optional test filtering.
"""

import sys
import unittest
import argparse
import time
from pathlib import Path

# Import all test modules
from test_pipeline import TestVisualCuePipeline, TestPipelineComponents
from test_dataset import TestPipelineDataset, TestImageItem
from test_adapters import (
    TestMaskGeneratorAdapter, TestImageProcessorAdapter, 
    TestCLIPAwayAdapter, TestEvaluatorAdapter, TestAdapterIntegration
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run Visual Cue Pipeline Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  pipeline    - End-to-end pipeline integration tests
  dataset     - Dataset discovery and management tests  
  adapters    - Component adapter integration tests
  components  - Individual component tests
  
Examples:
  python run_all_tests.py                    # Run all tests
  python run_all_tests.py --category dataset # Run only dataset tests
  python run_all_tests.py --verbose          # Run with detailed output
  python run_all_tests.py --fast             # Skip slow integration tests
        """
    )
    
    parser.add_argument(
        '--category', '-c',
        choices=['pipeline', 'dataset', 'adapters', 'components', 'all'],
        default='all',
        help='Test category to run (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose test output'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Skip slow integration tests'
    )
    
    parser.add_argument(
        '--failfast',
        action='store_true',
        help='Stop on first failure'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        help='Run only tests matching pattern'
    )
    
    return parser.parse_args()


def create_test_suite(category: str, fast: bool = False, pattern: str = None):
    """Create test suite based on category"""
    suite = unittest.TestSuite()
    
    # Define test classes by category
    test_classes = {
        'pipeline': [TestVisualCuePipeline],
        'dataset': [TestPipelineDataset, TestImageItem],
        'adapters': [
            TestMaskGeneratorAdapter, TestImageProcessorAdapter,
            TestCLIPAwayAdapter, TestEvaluatorAdapter, TestAdapterIntegration
        ],
        'components': [TestPipelineComponents],
        'all': [
            TestVisualCuePipeline, TestPipelineComponents,
            TestPipelineDataset, TestImageItem,
            TestMaskGeneratorAdapter, TestImageProcessorAdapter,
            TestCLIPAwayAdapter, TestEvaluatorAdapter, TestAdapterIntegration
        ]
    }
    
    # Get test classes for category
    classes_to_test = test_classes.get(category, test_classes['all'])
    
    # Add tests to suite
    for test_class in classes_to_test:
        # Skip slow tests if fast mode
        if fast and hasattr(test_class, '_slow_tests'):
            continue
        
        if pattern:
            # Filter tests by pattern
            for test_name in unittest.TestLoader().getTestCaseNames(test_class):
                if pattern.lower() in test_name.lower():
                    suite.addTest(test_class(test_name))
        else:
            # Add all tests from class
            suite.addTest(unittest.makeSuite(test_class))
    
    return suite


def run_tests(args):
    """Run the test suite with specified arguments"""
    print("Visual Cue Pipeline Test Suite")
    print("=" * 60)
    print(f"Category: {args.category}")
    print(f"Fast mode: {args.fast}")
    if args.pattern:
        print(f"Pattern filter: {args.pattern}")
    print("=" * 60)
    
    # Create test suite
    suite = create_test_suite(args.category, args.fast, args.pattern)
    
    # Configure test runner
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=args.failfast,
        stream=sys.stdout
    )
    
    # Run tests
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print detailed summary
    print("\n" + "=" * 60)
    print("TEST EXECUTION SUMMARY")
    print("=" * 60)
    
    duration = end_time - start_time
    print(f"Execution time: {duration:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    # Print failure details
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"{i:2d}. {test}")
            if args.verbose:
                print(f"     {traceback.strip()}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"{i:2d}. {test}")
            if args.verbose:
                print(f"     {traceback.strip()}")
    
    # Print recommendations
    if result.failures or result.errors:
        print(f"\nRECOMMENDATIONS:")
        print("• Run with --verbose for detailed error information")
        print("• Run individual test categories to isolate issues")
        print("• Check test dependencies and mock configurations")
    else:
        print(f"\n✅ All tests passed successfully!")
    
    print("=" * 60)
    
    return result.wasSuccessful()


def validate_test_environment():
    """Validate test environment and dependencies"""
    print("Validating test environment...")
    
    validation_issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        validation_issues.append("Python 3.8+ required")
    
    # Check required modules
    required_modules = ['unittest', 'tempfile', 'pathlib', 'json']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            validation_issues.append(f"Missing required module: {module}")
    
    # Check optional modules for enhanced testing
    optional_modules = ['PIL', 'numpy', 'yaml']
    missing_optional = []
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)
    
    if validation_issues:
        print("❌ Environment validation failed:")
        for issue in validation_issues:
            print(f"  • {issue}")
        return False
    
    if missing_optional:
        print("⚠️  Optional modules missing (some tests may be limited):")
        for module in missing_optional:
            print(f"  • {module}")
    
    print("✅ Environment validation passed")
    return True


def main():
    """Main test runner entry point"""
    args = parse_arguments()
    
    # Validate environment
    if not validate_test_environment():
        print("Fix environment issues before running tests")
        sys.exit(1)
    
    # Run tests
    try:
        success = run_tests(args)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\nUnexpected error during test execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
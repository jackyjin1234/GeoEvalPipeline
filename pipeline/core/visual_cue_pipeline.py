#!/usr/bin/env python3
"""
Visual Cue Evaluation Pipeline Orchestrator

This is the main pipeline orchestrator that coordinates all phases of the
visual cue evaluation process, from dataset discovery through final evaluation.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from .pipeline_config import PipelineConfig
from .pipeline_dataset import PipelineDataset, ImageItem
from .pipeline_state import PipelineState, PipelinePhase, PhaseStatus
from .pipeline_utils import (
    setup_logging, validate_environment, monitor_system_resources,
    create_progress_bar, format_duration, generate_run_summary
)

from adapters.mask_generator import MaskGeneratorAdapter
from adapters.image_processor import ImageProcessorAdapter
from adapters.clipaway_adapter import CLIPAwayAdapter
from adapters.evaluator_adapter import EvaluatorAdapter

logger = logging.getLogger(__name__)


class VisualCuePipeline:
    """
    Main pipeline orchestrator for visual cue evaluation.
    
    Coordinates all pipeline phases with optimized data flow, state management,
    and error handling to efficiently evaluate visual cue importance for
    geolocation tasks using the Cityscapes dataset.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: PipelineConfig object with all settings
        """
        self.config = config
        self.selected_items: List[ImageItem] = []
        
        # Initialize pipeline state
        self.state = PipelineState(
            config.resume.state_file,
            config.to_dict()
        )
        
        # Initialize components (lazy loading)
        self.dataset = None
        self.mask_generator = None
        self.image_processor = None
        self.clipaway_adapter = None
        self.evaluator_adapter = None
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.phase_timings: Dict[PipelinePhase, float] = {}
        
        logger.info("VisualCuePipeline initialized")
    
    def _initialize_components(self):
        """Initialize pipeline components (lazy loading)"""
        if self.dataset is None:
            self.dataset = PipelineDataset(str(self.config.dataset.cityscapes_root))
        
        if self.mask_generator is None:
            self.mask_generator = MaskGeneratorAdapter(
                target_features=self.config.features.target_features,
                combine_masks=self.config.features.combine_masks,
                mask_suffix=self.config.features.mask_suffix
            )
        
        if self.image_processor is None:
            self.image_processor = ImageProcessorAdapter(
                target_size=self.config.processing.image_size,
                downscale_interpolation=self.config.processing.downscale_interpolation,
                upscale_interpolation=self.config.processing.upscale_interpolation
            )
        
        if self.clipaway_adapter is None:
            self.clipaway_adapter = CLIPAwayAdapter(
                clipaway_path=self.config.clipaway.path,
                device=self.config.clipaway.device,
                strength=self.config.clipaway.strength,
                scale=self.config.clipaway.scale,
                seed=self.config.clipaway.seed,
                model_key=self.config.clipaway.model_key
            )
        
        if self.evaluator_adapter is None:
            self.evaluator_adapter = EvaluatorAdapter(
                model=self.config.evaluation.model,
                concurrent_requests=self.config.evaluation.concurrent_requests,
                cache_responses=self.config.evaluation.cache_responses,
                use_ground_truth=self.config.evaluation.use_ground_truth,
                mock_mode=(self.config.evaluation.model == "mock")
            )
        
        logger.info("Pipeline components initialized")
    
    async def run_pipeline(self, resume: bool = False) -> Dict[str, Any]:
        """
        Run the complete visual cue evaluation pipeline.
        
        Args:
            resume: Whether to resume from previous state
        
        Returns:
            Dictionary with pipeline execution results
        """
        self.start_time = datetime.now()
        logger.info(f"Starting visual cue evaluation pipeline - Resume: {resume}")
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Create output directories
            self.config.create_output_directories()
            
            # Show resume report if resuming
            if resume:
                logger.info("Resume Report:")
                logger.info(self.state.get_resume_report())
            
            # Execute pipeline phases
            await self._execute_dataset_discovery()
            await self._execute_image_selection()
            await self._execute_mask_generation()
            await self._execute_image_processing()
            await self._execute_clipaway_processing()
            await self._execute_evaluation()
            await self._execute_report_generation()
            
            # Complete pipeline
            self.state.complete_pipeline()
            
            # Generate final results
            results = self._generate_final_results()
            
            duration = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Pipeline completed successfully in {format_duration(duration)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
        
        finally:
            # Cleanup resources
            self._cleanup_resources()
    
    async def _execute_dataset_discovery(self):
        """Phase 1: Discover and validate dataset"""
        phase = PipelinePhase.DATASET_DISCOVERY
        
        if self.state.phases[phase].status == PhaseStatus.COMPLETED:
            logger.info("Dataset discovery already completed - skipping")
            return
        
        logger.info("Phase 1: Dataset Discovery")
        phase_start = time.time()
        self.state.start_phase(phase)
        
        try:
            # Discover all available triplets
            all_items = self.dataset.discover_all_triplets()
            
            # Update progress
            self.state.update_phase_progress(phase, len(all_items))
            
            # Get statistics
            stats = self.dataset.get_statistics()
            
            # Complete phase
            artifacts = {
                'total_items_found': len(all_items),
                'dataset_statistics': stats
            }
            self.state.complete_phase(phase, artifacts)
            
            # Log results
            logger.info(f"Discovered {len(all_items)} valid image triplets")
            logger.info(f"Cities: {stats['cities']}")
            logger.info(f"Items per city: {stats['items_per_city']}")
            
        except Exception as e:
            self.state.fail_phase(phase, str(e))
            raise
        
        finally:
            self.phase_timings[phase] = time.time() - phase_start
    
    async def _execute_image_selection(self):
        """Phase 2: Select images for processing"""
        phase = PipelinePhase.IMAGE_SELECTION
        
        if self.state.phases[phase].status == PhaseStatus.COMPLETED:
            logger.info("Image selection already completed - loading cached selection")
            # TODO: Load selected items from state
            return
        
        logger.info("Phase 2: Image Selection")
        phase_start = time.time()
        self.state.start_phase(phase, len(self.config.dataset.cities))
        
        try:
            # Select images per city
            self.selected_items = self.dataset.select_images_per_city(
                cities=self.config.dataset.cities,
                images_per_city=self.config.dataset.images_per_city,
                selection_method=self.config.dataset.selection_method
            )
            
            # Apply max_images limit if configured
            if self.config.evaluation.max_images:
                self.selected_items = self.selected_items[:self.config.evaluation.max_images]
                logger.info(f"Limited selection to {len(self.selected_items)} images")
            
            # Validate selection
            validation = self.dataset.validate_selection(self.selected_items)
            
            # Initialize pipeline state with selected items
            self.state.start_pipeline(self.selected_items)
            
            # Update progress
            self.state.update_phase_progress(phase, len(self.selected_items))
            
            # Complete phase
            artifacts = {
                'selected_items_count': len(self.selected_items),
                'selection_validation': validation
            }
            self.state.complete_phase(phase, artifacts)
            
            # Log results
            logger.info(f"Selected {len(self.selected_items)} images for processing")
            logger.info(f"Cities represented: {validation['cities_represented']}")
            
            if not validation['all_files_exist']:
                logger.warning(f"Some files missing: {validation['missing_files']}")
            
            if not validation['all_have_gps']:
                logger.warning(f"Some GPS data missing: {validation['missing_gps']}")
            
        except Exception as e:
            self.state.fail_phase(phase, str(e))
            raise
        
        finally:
            self.phase_timings[phase] = time.time() - phase_start
    
    async def _execute_mask_generation(self):
        """Phase 3: Generate binary masks"""
        phase = PipelinePhase.MASK_GENERATION
        
        # Check which items need mask generation
        items_needing_masks = [
            item for item in self.selected_items
            if phase not in self.state.items.get(item.image_id, type('', (), {'phases_completed': set()})()).phases_completed
        ]
        
        if not items_needing_masks:
            logger.info("Mask generation already completed - skipping")
            return
        
        logger.info(f"Phase 3: Mask Generation ({len(items_needing_masks)} items)")
        phase_start = time.time()
        self.state.start_phase(phase, len(items_needing_masks))
        
        try:
            # Generate masks in batches
            batch_size = self.config.processing.batch_size
            completed_items = 0
            
            for i in range(0, len(items_needing_masks), batch_size):
                batch = items_needing_masks[i:i + batch_size]
                
                logger.info(f"Processing mask batch {i//batch_size + 1} ({len(batch)} items)")
                
                # Generate masks for batch
                results = self.mask_generator.generate_masks_batch(
                    batch, 
                    self.config.get_masks_dir()
                )
                
                # Update state for completed items
                for item in batch:
                    if all(results[feature][batch.index(item)] for feature in self.config.features.target_features):
                        self.state.mark_item_phase_completed(item.image_id, phase)
                        completed_items += 1
                    else:
                        self.state.mark_item_phase_failed(item.image_id, phase, "Mask generation failed")
                
                # Update progress
                self.state.update_phase_progress(phase, completed_items)
                
                # Log progress
                progress_bar = create_progress_bar(completed_items, len(items_needing_masks))
                logger.info(f"Mask generation progress: {progress_bar}")
            
            # Validate results
            validation = self.mask_generator.validate_masks(items_needing_masks)
            
            # Complete phase
            artifacts = {
                'masks_generated': completed_items,
                'validation_report': validation
            }
            self.state.complete_phase(phase, artifacts)
            
            logger.info(f"Generated masks for {completed_items}/{len(items_needing_masks)} items")
            logger.info(f"Success rate: {validation['success_rate']:.1f}%")
            
        except Exception as e:
            self.state.fail_phase(phase, str(e))
            raise
        
        finally:
            self.phase_timings[phase] = time.time() - phase_start
    
    async def _execute_image_processing(self):
        """Phase 4: Process images (split and downscale)"""
        phase = PipelinePhase.IMAGE_PROCESSING
        
        # Check which items need processing
        items_needing_processing = [
            item for item in self.selected_items
            if phase not in self.state.items.get(item.image_id, type('', (), {'phases_completed': set()})()).phases_completed
        ]
        
        if not items_needing_processing:
            logger.info("Image processing already completed - skipping")
            return
        
        logger.info(f"Phase 4: Image Processing ({len(items_needing_processing)} items)")
        phase_start = time.time()
        self.state.start_phase(phase, len(items_needing_processing))
        
        try:
            # Process images in batches
            batch_size = self.config.processing.batch_size
            completed_items = 0
            
            for i in range(0, len(items_needing_processing), batch_size):
                batch = items_needing_processing[i:i + batch_size]
                
                logger.info(f"Processing image batch {i//batch_size + 1} ({len(batch)} items)")
                
                # Process batch
                results = self.image_processor.process_images_batch(
                    batch,
                    self.config.get_processed_dir(),
                    process_masks=True
                )
                
                # Update state for completed items
                for j, item in enumerate(batch):
                    if (results['original_left'][j] and results['original_right'][j] and
                        all(results['mask_results'].get(feature, [None])[j] for feature in self.config.features.target_features)):
                        self.state.mark_item_phase_completed(item.image_id, phase)
                        completed_items += 1
                    else:
                        self.state.mark_item_phase_failed(item.image_id, phase, "Image processing failed")
                
                # Update progress
                self.state.update_phase_progress(phase, completed_items)
                
                # Log progress
                progress_bar = create_progress_bar(completed_items, len(items_needing_processing))
                logger.info(f"Image processing progress: {progress_bar}")
            
            # Validate results
            validation = self.image_processor.validate_processed_images(items_needing_processing)
            
            # Complete phase
            artifacts = {
                'images_processed': completed_items,
                'validation_report': validation
            }
            self.state.complete_phase(phase, artifacts)
            
            logger.info(f"Processed {completed_items}/{len(items_needing_processing)} items")
            logger.info(f"Success rate: {validation['original_success_rate']:.1f}%")
            
        except Exception as e:
            self.state.fail_phase(phase, str(e))
            raise
        
        finally:
            self.phase_timings[phase] = time.time() - phase_start
    
    async def _execute_clipaway_processing(self):
        """Phase 5: CLIPAway visual cue removal"""
        phase = PipelinePhase.CLIPAWAY_PROCESSING
        
        # Check which items need CLIPAway processing
        items_needing_clipaway = [
            item for item in self.selected_items
            if phase not in self.state.items.get(item.image_id, type('', (), {'phases_completed': set()})()).phases_completed
        ]
        
        if not items_needing_clipaway:
            logger.info("CLIPAway processing already completed - skipping")
            return
        
        logger.info(f"Phase 5: CLIPAway Processing ({len(items_needing_clipaway)} items)")
        phase_start = time.time()
        self.state.start_phase(phase, len(items_needing_clipaway))
        
        try:
            # Process items in smaller batches (CLIPAway is memory intensive)
            batch_size = min(self.config.processing.batch_size, 3)  # Smaller batches for GPU memory
            completed_items = 0
            
            for i in range(0, len(items_needing_clipaway), batch_size):
                batch = items_needing_clipaway[i:i + batch_size]
                
                logger.info(f"Processing CLIPAway batch {i//batch_size + 1} ({len(batch)} items)")
                
                # Process batch
                results = self.clipaway_adapter.process_items_batch(
                    batch,
                    self.config.features.target_features,
                    self.config.get_clipaway_dir()
                )
                
                # Update state for completed items
                for j, item in enumerate(batch):
                    if all(results[feature][j] for feature in self.config.features.target_features):
                        self.state.mark_item_phase_completed(item.image_id, phase)
                        completed_items += 1
                    else:
                        self.state.mark_item_phase_failed(item.image_id, phase, "CLIPAway processing failed")
                
                # Update progress
                self.state.update_phase_progress(phase, completed_items)
                
                # Log progress with memory info
                progress_bar = create_progress_bar(completed_items, len(items_needing_clipaway))
                resources = monitor_system_resources()
                logger.info(f"CLIPAway progress: {progress_bar} (GPU: {resources['gpu_memory_percent']:.1f}%)")
            
            # Validate results
            validation = self.clipaway_adapter.validate_results(
                items_needing_clipaway,
                self.config.features.target_features
            )
            
            # Complete phase
            artifacts = {
                'items_processed': completed_items,
                'validation_report': validation
            }
            self.state.complete_phase(phase, artifacts)
            
            logger.info(f"CLIPAway processed {completed_items}/{len(items_needing_clipaway)} items")
            logger.info(f"Success rate: {validation['success_rate']:.1f}%")
            
        except Exception as e:
            self.state.fail_phase(phase, str(e))
            raise
        
        finally:
            self.phase_timings[phase] = time.time() - phase_start
    
    async def _execute_evaluation(self):
        """Phase 6: Geolocation evaluation"""
        phase = PipelinePhase.EVALUATION
        
        # Check which items need evaluation
        items_needing_evaluation = [
            item for item in self.selected_items
            if phase not in self.state.items.get(item.image_id, type('', (), {'phases_completed': set()})()).phases_completed
        ]
        
        if not items_needing_evaluation:
            logger.info("Evaluation already completed - skipping")
            return
        
        logger.info(f"Phase 6: Geolocation Evaluation ({len(items_needing_evaluation)} items)")
        phase_start = time.time()
        self.state.start_phase(phase, len(items_needing_evaluation))
        
        try:
            # Run evaluation
            evaluation_results = await self.evaluator_adapter.evaluate_items_batch(
                items_needing_evaluation,
                self.config.features.target_features,
                self.config.get_evaluation_dir()
            )
            
            # Update state for all items (evaluation is all-or-nothing per item)
            completed_items = evaluation_results['completed_evaluations'] // (1 + len(self.config.features.target_features))
            for item in items_needing_evaluation[:completed_items]:
                self.state.mark_item_phase_completed(item.image_id, phase)
            
            # Update progress
            self.state.update_phase_progress(phase, completed_items)
            
            # Complete phase
            artifacts = {
                'evaluation_results': evaluation_results['summary_statistics'],
                'total_evaluations': evaluation_results['completed_evaluations']
            }
            self.state.complete_phase(phase, artifacts)
            
            logger.info(f"Completed {evaluation_results['completed_evaluations']} evaluations")
            logger.info(f"Items evaluated: {completed_items}/{len(items_needing_evaluation)}")
            
        except Exception as e:
            self.state.fail_phase(phase, str(e))
            raise
        
        finally:
            self.phase_timings[phase] = time.time() - phase_start
    
    async def _execute_report_generation(self):
        """Phase 7: Generate final reports"""
        phase = PipelinePhase.REPORT_GENERATION
        
        if self.state.phases[phase].status == PhaseStatus.COMPLETED:
            logger.info("Report generation already completed - skipping")
            return
        
        logger.info("Phase 7: Report Generation")
        phase_start = time.time()
        self.state.start_phase(phase, 1)
        
        try:
            # Generate comprehensive report
            report_data = self._compile_pipeline_report()
            
            # Save report to file
            report_file = self.config.output.base_directory / "pipeline_report.json"
            import json
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Save configuration used
            self.config.save_config()
            
            # Update progress
            self.state.update_phase_progress(phase, 1)
            
            # Complete phase
            artifacts = {
                'report_file': str(report_file),
                'report_summary': report_data['summary']
            }
            self.state.complete_phase(phase, artifacts)
            
            logger.info(f"Pipeline report saved to: {report_file}")
            
        except Exception as e:
            self.state.fail_phase(phase, str(e))
            raise
        
        finally:
            self.phase_timings[phase] = time.time() - phase_start
    
    def _compile_pipeline_report(self) -> Dict[str, Any]:
        """Compile comprehensive pipeline execution report"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'pipeline_info': {
                'version': '1.0.0',
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'configuration': self.config.to_dict()
            },
            'execution_summary': {
                'total_items_selected': len(self.selected_items),
                'phases_completed': len([p for p in self.state.phases.values() if p.status == PhaseStatus.COMPLETED]),
                'total_phases': len(PipelinePhase),
                'overall_success_rate': self._calculate_overall_success_rate()
            },
            'phase_timings': {
                phase.value: duration for phase, duration in self.phase_timings.items()
            },
            'phase_details': {
                phase.value: {
                    'status': state.status.value,
                    'duration': state.get_duration(),
                    'progress_percentage': state.get_progress_percentage(),
                    'artifacts': state.artifacts
                }
                for phase, state in self.state.phases.items()
            },
            'system_info': validate_environment(),
            'performance_metrics': self._gather_performance_metrics(),
            'summary': self._generate_executive_summary()
        }
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall pipeline success rate"""
        if not self.selected_items:
            return 0.0
        
        total_phases = len(PipelinePhase)
        successful_phases = 0
        
        for item_id in [item.image_id for item in self.selected_items]:
            item_state = self.state.items.get(item_id)
            if item_state:
                successful_phases += len(item_state.phases_completed)
        
        return (successful_phases / (len(self.selected_items) * total_phases)) * 100
    
    def _gather_performance_metrics(self) -> Dict[str, Any]:
        """Gather performance metrics from pipeline execution"""
        return {
            'total_processing_time': sum(self.phase_timings.values()),
            'average_time_per_item': sum(self.phase_timings.values()) / len(self.selected_items) if self.selected_items else 0,
            'phase_time_distribution': {
                phase.value: (duration / sum(self.phase_timings.values()) * 100) 
                for phase, duration in self.phase_timings.items()
            } if self.phase_timings else {},
            'system_resources': monitor_system_resources()
        }
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary of pipeline execution"""
        if not self.selected_items:
            return "Pipeline execution failed - no items processed"
        
        total_duration = sum(self.phase_timings.values())
        success_rate = self._calculate_overall_success_rate()
        
        summary_lines = [
            f"Visual Cue Evaluation Pipeline completed successfully",
            f"Processed {len(self.selected_items)} images across {len(self.config.dataset.cities)} cities",
            f"Evaluated {len(self.config.features.target_features)} visual features: {', '.join(self.config.features.target_features)}",
            f"Overall success rate: {success_rate:.1f}%",
            f"Total processing time: {format_duration(total_duration)}",
            f"Average time per item: {format_duration(total_duration / len(self.selected_items))}"
        ]
        
        return " | ".join(summary_lines)
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate final pipeline results"""
        return {
            'success': True,
            'total_items_processed': len(self.selected_items),
            'execution_time': sum(self.phase_timings.values()),
            'success_rate': self._calculate_overall_success_rate(),
            'output_directory': str(self.config.output.base_directory),
            'phase_timings': {phase.value: duration for phase, duration in self.phase_timings.items()},
            'summary': self._generate_executive_summary()
        }
    
    def _cleanup_resources(self):
        """Clean up pipeline resources"""
        try:
            if self.clipaway_adapter:
                self.clipaway_adapter.cleanup_models()
            
            logger.info("Pipeline resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def get_progress_report(self) -> str:
        """Get current pipeline progress as human-readable report"""
        return self.state.get_resume_report()
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate pipeline setup before execution"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate environment
        env_validation = validate_environment()
        if not env_validation['valid']:
            validation['valid'] = False
            validation['errors'].extend(env_validation['errors'])
        validation['warnings'].extend(env_validation['warnings'])
        
        # Validate configuration
        try:
            self.config._validate_config()
        except ValueError as e:
            validation['valid'] = False
            validation['errors'].append(f"Configuration error: {e}")
        
        # Validate dataset access
        try:
            if self.dataset is None:
                self.dataset = PipelineDataset(str(self.config.dataset.cityscapes_root))
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"Dataset access error: {e}")
        
        return validation


if __name__ == "__main__":
    # Quick test of pipeline orchestrator
    import sys
    import tempfile
    
    if len(sys.argv) != 2:
        print("Usage: python visual_cue_pipeline.py <config_file>")
        sys.exit(1)
    
    # Setup logging
    setup_logging("INFO")
    
    try:
        # Load configuration
        config = PipelineConfig(sys.argv[1])
        
        # Create pipeline
        pipeline = VisualCuePipeline(config)
        
        # Validate setup
        validation = pipeline.validate_setup()
        if not validation['valid']:
            print("Pipeline validation failed:")
            for error in validation['errors']:
                print(f"  ERROR: {error}")
            sys.exit(1)
        
        if validation['warnings']:
            print("Pipeline validation warnings:")
            for warning in validation['warnings']:
                print(f"  WARNING: {warning}")
        
        print("Pipeline validation passed - ready for execution")
        
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        sys.exit(1)
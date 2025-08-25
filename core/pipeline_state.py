#!/usr/bin/env python3
"""
Simple Pipeline Execution Tracking

This module handles basic pipeline execution tracking and statistics.
"""

import logging
from datetime import datetime
from typing import Dict, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PhaseStatus(Enum):
    """Pipeline phase execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelinePhase(Enum):
    """Pipeline execution phases"""
    DATASET_DISCOVERY = "dataset_discovery"
    IMAGE_SELECTION = "image_selection"
    MASK_GENERATION = "mask_generation"
    IMAGE_PROCESSING = "image_processing"
    CLIPAWAY_PROCESSING = "clipaway_processing"
    EVALUATION = "evaluation"
    REPORT_GENERATION = "report_generation"


@dataclass
class PhaseState:
    """Simple state tracking for individual pipeline phase"""
    phase: PipelinePhase
    status: PhaseStatus = PhaseStatus.PENDING
    start_time: datetime = None
    end_time: datetime = None
    progress: int = 0
    total: int = 0
    error_message: str = None
    
    def start(self, total_items: int = 0):
        """Mark phase as started"""
        self.status = PhaseStatus.IN_PROGRESS
        self.start_time = datetime.now()
        self.total = total_items
        self.progress = 0
        self.error_message = None
    
    def update_progress(self, completed_items: int):
        """Update progress counter"""
        self.progress = completed_items
    
    def complete(self):
        """Mark phase as completed"""
        self.status = PhaseStatus.COMPLETED
        self.end_time = datetime.now()
        self.progress = self.total
    
    def fail(self, error_message: str):
        """Mark phase as failed"""
        self.status = PhaseStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error_message
    
    def get_duration(self) -> float:
        """Get phase duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class PipelineState:
    """
    Simple pipeline execution tracker.
    
    Tracks overall pipeline progress and phase states for monitoring only.
    """
    
    def __init__(self):
        """
        Initialize simple pipeline state tracker.
        """
        # Pipeline execution metadata
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time: datetime = None
        self.end_time: datetime = None
        
        # Phase tracking
        self.phases: Dict[PipelinePhase, PhaseState] = {
            phase: PhaseState(phase) for phase in PipelinePhase
        }
        
        # Simple statistics
        self.total_items: int = 0
        
        logger.info(f"Pipeline state tracker initialized (run_id: {self.run_id})")
    
    def start_pipeline(self, total_items: int):
        """
        Initialize pipeline execution.
        
        Args:
            total_items: Number of items to be processed
        """
        self.start_time = datetime.now()
        self.total_items = total_items
        
        logger.info(f"Pipeline started with {total_items} items")
    
    def start_phase(self, phase: PipelinePhase, total_items: int = 0):
        """Start a pipeline phase"""
        self.phases[phase].start(total_items)
        logger.info(f"Started phase: {phase.value}")
    
    def update_phase_progress(self, phase: PipelinePhase, completed_items: int):
        """Update progress for a phase"""
        self.phases[phase].update_progress(completed_items)
    
    def complete_phase(self, phase: PipelinePhase):
        """Mark a phase as completed"""
        self.phases[phase].complete()
        logger.info(f"Completed phase: {phase.value}")
    
    def fail_phase(self, phase: PipelinePhase, error_message: str):
        """Mark a phase as failed"""
        self.phases[phase].fail(error_message)
        logger.error(f"Phase {phase.value} failed: {error_message}")
    
    def complete_pipeline(self):
        """Mark pipeline as completed"""
        self.end_time = datetime.now()
        
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        logger.info(f"Pipeline completed in {duration:.1f} seconds")

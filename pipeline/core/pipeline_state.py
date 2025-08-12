#!/usr/bin/env python3
"""
Pipeline State Management and Resume Functionality

This module handles tracking pipeline execution state, saving checkpoints,
and enabling resume functionality for long-running pipeline executions.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field, asdict
from copy import deepcopy

from .pipeline_dataset import ImageItem

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
    """State tracking for individual pipeline phase"""
    phase: PipelinePhase
    status: PhaseStatus = PhaseStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: int = 0  # Number of items processed
    total: int = 0  # Total items to process
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
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
    
    def complete(self, artifacts: Optional[Dict[str, Any]] = None):
        """Mark phase as completed"""
        self.status = PhaseStatus.COMPLETED
        self.end_time = datetime.now()
        self.progress = self.total
        if artifacts:
            self.artifacts.update(artifacts)
    
    def fail(self, error_message: str):
        """Mark phase as failed"""
        self.status = PhaseStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error_message
    
    def skip(self, reason: str):
        """Mark phase as skipped"""
        self.status = PhaseStatus.SKIPPED
        self.end_time = datetime.now()
        self.error_message = reason
    
    def get_duration(self) -> Optional[float]:
        """Get phase duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage"""
        if self.total == 0:
            return 0.0
        return (self.progress / self.total) * 100


@dataclass
class ItemState:
    """State tracking for individual image item processing"""
    image_id: str
    phases_completed: Set[PipelinePhase] = field(default_factory=set)
    phases_failed: Set[PipelinePhase] = field(default_factory=set)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error_messages: Dict[PipelinePhase, str] = field(default_factory=dict)
    
    def is_phase_completed(self, phase: PipelinePhase) -> bool:
        """Check if phase is completed for this item"""
        return phase in self.phases_completed
    
    def is_phase_failed(self, phase: PipelinePhase) -> bool:
        """Check if phase failed for this item"""
        return phase in self.phases_failed
    
    def mark_phase_completed(self, phase: PipelinePhase, artifacts: Optional[Dict[str, Any]] = None):
        """Mark phase as completed for this item"""
        self.phases_completed.add(phase)
        self.phases_failed.discard(phase)
        if artifacts:
            self.artifacts.update(artifacts)
    
    def mark_phase_failed(self, phase: PipelinePhase, error_message: str):
        """Mark phase as failed for this item"""
        self.phases_failed.add(phase)
        self.phases_completed.discard(phase)
        self.error_messages[phase] = error_message
    
    def get_completed_phases(self) -> List[PipelinePhase]:
        """Get list of completed phases"""
        return list(self.phases_completed)
    
    def get_failed_phases(self) -> List[PipelinePhase]:
        """Get list of failed phases"""
        return list(self.phases_failed)


class PipelineState:
    """
    Main pipeline state manager with checkpointing and resume capabilities.
    
    Tracks overall pipeline progress, individual phase states, and per-item
    processing status to enable robust resume functionality.
    """
    
    def __init__(self, state_file: Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline state manager.
        
        Args:
            state_file: Path to state persistence file
            config: Pipeline configuration for validation
        """
        self.state_file = Path(state_file)
        self.config = config or {}
        
        # Pipeline execution metadata
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Phase tracking
        self.phases: Dict[PipelinePhase, PhaseState] = {
            phase: PhaseState(phase) for phase in PipelinePhase
        }
        
        # Item tracking
        self.items: Dict[str, ItemState] = {}
        self.selected_items: List[str] = []  # Ordered list of selected image IDs
        
        # Statistics
        self.stats: Dict[str, Any] = {
            'total_items': 0,
            'items_completed': 0,
            'items_failed': 0,
            'phases_completed': 0,
            'total_phases': len(PipelinePhase),
            'checkpoints_saved': 0,
            'last_checkpoint': None
        }
        
        # Load existing state if available
        self._load_state()
        
        logger.info(f"Pipeline state manager initialized (run_id: {self.run_id})")
    
    def _load_state(self):
        """Load existing state from file if available"""
        if not self.state_file.exists():
            logger.info("No existing state file found, starting fresh")
            return
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            # Restore basic metadata
            self.run_id = data.get('run_id', self.run_id)
            self.start_time = datetime.fromisoformat(data['start_time']) if data.get('start_time') else None
            self.end_time = datetime.fromisoformat(data['end_time']) if data.get('end_time') else None
            
            # Restore phase states
            for phase_name, phase_data in data.get('phases', {}).items():
                try:
                    phase = PipelinePhase(phase_name)
                    phase_state = PhaseState(phase)
                    phase_state.status = PhaseStatus(phase_data['status'])
                    phase_state.start_time = datetime.fromisoformat(phase_data['start_time']) if phase_data.get('start_time') else None
                    phase_state.end_time = datetime.fromisoformat(phase_data['end_time']) if phase_data.get('end_time') else None
                    phase_state.progress = phase_data.get('progress', 0)
                    phase_state.total = phase_data.get('total', 0)
                    phase_state.error_message = phase_data.get('error_message')
                    phase_state.artifacts = phase_data.get('artifacts', {})
                    
                    self.phases[phase] = phase_state
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to restore phase state for {phase_name}: {e}")
            
            # Restore item states
            for item_id, item_data in data.get('items', {}).items():
                try:
                    item_state = ItemState(item_id)
                    item_state.phases_completed = set(PipelinePhase(p) for p in item_data.get('phases_completed', []))
                    item_state.phases_failed = set(PipelinePhase(p) for p in item_data.get('phases_failed', []))
                    item_state.artifacts = item_data.get('artifacts', {})
                    item_state.error_messages = {
                        PipelinePhase(k): v for k, v in item_data.get('error_messages', {}).items()
                    }
                    
                    self.items[item_id] = item_state
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to restore item state for {item_id}: {e}")
            
            # Restore other data
            self.selected_items = data.get('selected_items', [])
            self.stats = data.get('stats', self.stats)
            
            logger.info(f"Pipeline state restored from {self.state_file}")
            logger.info(f"Restored state for {len(self.items)} items across {len(self.phases)} phases")
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load state from {self.state_file}: {e}")
            logger.info("Starting with fresh state")
    
    def start_pipeline(self, selected_items: List[ImageItem]):
        """
        Initialize pipeline execution with selected items.
        
        Args:
            selected_items: List of selected ImageItem objects
        """
        self.start_time = datetime.now()
        self.selected_items = [item.image_id for item in selected_items]
        
        # Initialize item states
        for item in selected_items:
            if item.image_id not in self.items:
                self.items[item.image_id] = ItemState(item.image_id)
        
        # Update statistics
        self.stats['total_items'] = len(selected_items)
        
        self._save_checkpoint()
        logger.info(f"Pipeline started with {len(selected_items)} items")
    
    def start_phase(self, phase: PipelinePhase, total_items: int = 0):
        """Start a pipeline phase"""
        self.phases[phase].start(total_items)
        self._save_checkpoint()
        logger.info(f"Started phase: {phase.value}")
    
    def update_phase_progress(self, phase: PipelinePhase, completed_items: int):
        """Update progress for a phase"""
        self.phases[phase].update_progress(completed_items)
        
        # Update overall statistics
        self._update_stats()
    
    def complete_phase(self, phase: PipelinePhase, artifacts: Optional[Dict[str, Any]] = None):
        """Mark a phase as completed"""
        self.phases[phase].complete(artifacts)
        self.stats['phases_completed'] = sum(1 for p in self.phases.values() if p.status == PhaseStatus.COMPLETED)
        
        self._save_checkpoint()
        logger.info(f"Completed phase: {phase.value}")
    
    def fail_phase(self, phase: PipelinePhase, error_message: str):
        """Mark a phase as failed"""
        self.phases[phase].fail(error_message)
        self._save_checkpoint()
        logger.error(f"Phase {phase.value} failed: {error_message}")
    
    def skip_phase(self, phase: PipelinePhase, reason: str):
        """Mark a phase as skipped"""
        self.phases[phase].skip(reason)
        self._save_checkpoint()
        logger.info(f"Skipped phase {phase.value}: {reason}")
    
    def mark_item_phase_completed(self, item_id: str, phase: PipelinePhase, 
                                 artifacts: Optional[Dict[str, Any]] = None):
        """Mark a phase as completed for specific item"""
        if item_id not in self.items:
            self.items[item_id] = ItemState(item_id)
        
        self.items[item_id].mark_phase_completed(phase, artifacts)
        self._update_stats()
    
    def mark_item_phase_failed(self, item_id: str, phase: PipelinePhase, error_message: str):
        """Mark a phase as failed for specific item"""
        if item_id not in self.items:
            self.items[item_id] = ItemState(item_id)
        
        self.items[item_id].mark_phase_failed(phase, error_message)
        self._update_stats()
    
    def get_items_for_phase(self, phase: PipelinePhase) -> List[str]:
        """Get list of items that need processing for given phase"""
        items_needing_processing = []
        
        for item_id in self.selected_items:
            item_state = self.items.get(item_id)
            if not item_state or not item_state.is_phase_completed(phase):
                items_needing_processing.append(item_id)
        
        return items_needing_processing
    
    def get_completed_items_for_phase(self, phase: PipelinePhase) -> List[str]:
        """Get list of items that have completed given phase"""
        completed_items = []
        
        for item_id in self.selected_items:
            item_state = self.items.get(item_id)
            if item_state and item_state.is_phase_completed(phase):
                completed_items.append(item_id)
        
        return completed_items
    
    def is_phase_resumable(self, phase: PipelinePhase) -> bool:
        """Check if a phase can be resumed (has partial progress)"""
        phase_state = self.phases[phase]
        return (
            phase_state.status in [PhaseStatus.IN_PROGRESS, PhaseStatus.FAILED] and
            phase_state.progress > 0
        )
    
    def get_pipeline_progress(self) -> Dict[str, Any]:
        """Get overall pipeline progress information"""
        total_work = len(self.selected_items) * len(PipelinePhase)
        completed_work = sum(
            len(item_state.phases_completed) 
            for item_state in self.items.values()
        )
        
        progress_percentage = (completed_work / total_work * 100) if total_work > 0 else 0
        
        return {
            'total_items': self.stats['total_items'],
            'items_completed': self.stats['items_completed'],
            'items_failed': self.stats['items_failed'],
            'phases_completed': self.stats['phases_completed'],
            'total_phases': self.stats['total_phases'],
            'overall_progress_percentage': progress_percentage,
            'estimated_time_remaining': self._estimate_time_remaining(),
            'phase_progress': {
                phase.value: {
                    'status': state.status.value,
                    'progress_percentage': state.get_progress_percentage(),
                    'duration': state.get_duration()
                }
                for phase, state in self.phases.items()
            }
        }
    
    def _update_stats(self):
        """Update internal statistics"""
        self.stats['items_completed'] = sum(
            1 for item_state in self.items.values()
            if len(item_state.phases_completed) == len(PipelinePhase)
        )
        
        self.stats['items_failed'] = sum(
            1 for item_state in self.items.values()
            if item_state.phases_failed
        )
    
    def _estimate_time_remaining(self) -> Optional[float]:
        """Estimate remaining execution time in seconds"""
        if not self.start_time:
            return None
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        total_work = len(self.selected_items) * len(PipelinePhase)
        completed_work = sum(
            len(item_state.phases_completed) 
            for item_state in self.items.values()
        )
        
        if completed_work == 0:
            return None
        
        remaining_work = total_work - completed_work
        rate = completed_work / elapsed
        
        return remaining_work / rate if rate > 0 else None
    
    def _save_checkpoint(self):
        """Save current state to file"""
        try:
            # Ensure parent directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            data = {
                'run_id': self.run_id,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'selected_items': self.selected_items,
                'phases': {
                    phase.value: {
                        'status': state.status.value,
                        'start_time': state.start_time.isoformat() if state.start_time else None,
                        'end_time': state.end_time.isoformat() if state.end_time else None,
                        'progress': state.progress,
                        'total': state.total,
                        'error_message': state.error_message,
                        'artifacts': state.artifacts
                    }
                    for phase, state in self.phases.items()
                },
                'items': {
                    item_id: {
                        'phases_completed': [p.value for p in state.phases_completed],
                        'phases_failed': [p.value for p in state.phases_failed],
                        'artifacts': state.artifacts,
                        'error_messages': {k.value: v for k, v in state.error_messages.items()}
                    }
                    for item_id, state in self.items.items()
                },
                'stats': self.stats,
                'config_hash': hash(str(self.config))  # For validation
            }
            
            # Write to temporary file first, then rename for atomicity
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.rename(self.state_file)
            
            self.stats['checkpoints_saved'] += 1
            self.stats['last_checkpoint'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def complete_pipeline(self):
        """Mark pipeline as completed"""
        self.end_time = datetime.now()
        self._save_checkpoint()
        
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        logger.info(f"Pipeline completed in {duration:.1f} seconds")
    
    def get_resume_report(self) -> str:
        """Generate a human-readable resume report"""
        if not self.start_time:
            return "No previous execution found."
        
        progress = self.get_pipeline_progress()
        
        report = [
            f"Resume Report (Run ID: {self.run_id})",
            f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Progress: {progress['overall_progress_percentage']:.1f}%",
            f"Items: {progress['items_completed']}/{progress['total_items']} completed",
            f"Phases: {progress['phases_completed']}/{progress['total_phases']} completed",
            "",
            "Phase Status:"
        ]
        
        for phase, info in progress['phase_progress'].items():
            status_icon = "✓" if info['status'] == 'completed' else "⏳" if info['status'] == 'in_progress' else "○"
            report.append(f"  {status_icon} {phase}: {info['status']} ({info['progress_percentage']:.1f}%)")
        
        if progress['estimated_time_remaining']:
            report.append(f"\nEstimated time remaining: {progress['estimated_time_remaining']:.1f} seconds")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test state management
    import tempfile
    
    # Create temporary state file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        state_file = Path(f.name)
    
    try:
        # Test basic state operations
        state = PipelineState(state_file)
        
        # Simulate pipeline execution
        from .pipeline_dataset import ImageItem
        dummy_items = [
            ImageItem("test1", "city1", Path("test1.png"), Path("test1.json"), Path("test1_vehicle.json")),
            ImageItem("test2", "city1", Path("test2.png"), Path("test2.json"), Path("test2_vehicle.json"))
        ]
        
        state.start_pipeline(dummy_items)
        
        # Simulate phase execution
        state.start_phase(PipelinePhase.DATASET_DISCOVERY, 2)
        state.mark_item_phase_completed("test1", PipelinePhase.DATASET_DISCOVERY)
        state.mark_item_phase_completed("test2", PipelinePhase.DATASET_DISCOVERY)
        state.complete_phase(PipelinePhase.DATASET_DISCOVERY)
        
        print("State management test completed successfully")
        print(state.get_resume_report())
        
    finally:
        # Cleanup
        if state_file.exists():
            state_file.unlink()
#!/usr/bin/env python3
"""
Pipeline Dataset Management

This module handles dataset discovery, validation, and systematic selection
of images for the visual cue evaluation pipeline.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ImageItem:
    """
    Data structure representing a complete image processing item
    with all associated files and metadata.
    """
    image_id: str
    city: str
    image_path: Path
    annotation_path: Path
    gps_path: Path
    ground_truth_coords: Optional[Tuple[float, float]] = None
    
    # Processing artifacts (populated during pipeline execution)
    masks: Dict[str, Path] = field(default_factory=dict)
    processed_images: Dict[str, Dict[str, Path]] = field(default_factory=dict)
    split_images: Dict[str, Dict[str, Path]] = field(default_factory=dict)
    clipaway_results: Dict[str, Path] = field(default_factory=dict)
    evaluation_results: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and extract ground truth coordinates"""
        if self.gps_path.exists():
            self.ground_truth_coords = self._extract_gps_coordinates()
        else:
            logger.warning(f"GPS file not found for {self.image_id}: {self.gps_path}")
    
    def _extract_gps_coordinates(self) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from vehicle JSON file"""
        try:
            with open(self.gps_path, 'r') as f:
                data = json.load(f)
            
            lat = data.get('gpsLatitude')
            lon = data.get('gpsLongitude')
            
            if lat is not None and lon is not None:
                return (float(lat), float(lon))
            else:
                logger.warning(f"GPS coordinates missing in {self.gps_path}")
                return None
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error extracting GPS from {self.gps_path}: {e}")
            return None
    
    def is_valid(self) -> bool:
        """Check if all required files exist and GPS coordinates are available"""
        return (
            self.image_path.exists() and
            self.annotation_path.exists() and 
            self.gps_path.exists() and
            self.ground_truth_coords is not None
        )
    
    def get_base_filename(self) -> str:
        """Get base filename without extension"""
        return self.image_path.stem.replace('_leftImg8bit', '')


class PipelineDataset:
    """
    Dataset discovery and management for the visual cue evaluation pipeline.
    
    Handles systematic discovery of image/annotation/GPS triplets and 
    provides deterministic selection algorithms.
    """
    
    def __init__(self, cityscapes_root: str):
        self.cityscapes_root = Path(cityscapes_root)
        self.images_dir = self.cityscapes_root / "leftImg8bit" / "train"
        self.annotations_dir = self.cityscapes_root / "gtFine" / "train" 
        self.gps_dir = self.cityscapes_root / "vehicle" / "train"
        
        self.discovered_items: List[ImageItem] = []
        self.items_by_city: Dict[str, List[ImageItem]] = defaultdict(list)
        
        # Validate directory structure
        self._validate_directory_structure()
    
    def _validate_directory_structure(self):
        """Validate that all required directories exist"""
        required_dirs = [self.images_dir, self.annotations_dir, self.gps_dir]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise ValueError(f"Required directory not found: {dir_path}")
            
        logger.info(f"Dataset directory structure validated at: {self.cityscapes_root}")
    
    def discover_all_triplets(self) -> List[ImageItem]:
        """
        Discover all valid image/annotation/GPS triplets across all cities.
        
        Returns:
            List of ImageItem objects for all discovered valid triplets
        """
        logger.info("Starting dataset discovery...")
        
        self.discovered_items = []
        self.items_by_city = defaultdict(list)
        
        # Find all cities
        cities = [d.name for d in self.images_dir.iterdir() if d.is_dir()]
        logger.info(f"Found cities: {cities}")
        
        total_found = 0
        total_valid = 0
        
        for city in cities:
            city_items = self._discover_city_triplets(city)
            valid_items = [item for item in city_items if item.is_valid()]
            
            self.discovered_items.extend(valid_items)
            self.items_by_city[city] = valid_items
            
            total_found += len(city_items)
            total_valid += len(valid_items)
            
            logger.info(f"City {city}: {len(city_items)} found, {len(valid_items)} valid")
        
        logger.info(f"Dataset discovery completed: {total_found} triplets found, {total_valid} valid")
        return self.discovered_items
    
    def _discover_city_triplets(self, city: str) -> List[ImageItem]:
        """Discover all triplets for a specific city"""
        city_items = []
        
        city_images_dir = self.images_dir / city
        city_annotations_dir = self.annotations_dir / city
        city_gps_dir = self.gps_dir / city
        
        if not all(d.exists() for d in [city_images_dir, city_annotations_dir, city_gps_dir]):
            logger.warning(f"Incomplete directory structure for city: {city}")
            return city_items
        
        # Find all images in city
        image_files = list(city_images_dir.glob("*_leftImg8bit.png"))
        
        for image_path in image_files:
            # Extract base name and construct paths for annotation and GPS files
            base_name = image_path.stem.replace('_leftImg8bit', '')
            image_id = base_name
            
            annotation_path = city_annotations_dir / f"{base_name}_gtFine_polygons.json"
            gps_path = city_gps_dir / f"{base_name}_vehicle.json"
            
            # Create ImageItem
            item = ImageItem(
                image_id=image_id,
                city=city,
                image_path=image_path,
                annotation_path=annotation_path,
                gps_path=gps_path
            )
            
            city_items.append(item)
        
        return city_items
    
    def select_images_per_city(self, cities: List[str], images_per_city: int, 
                              selection_method: str = "systematic") -> List[ImageItem]:
        """
        Select specified number of images per city using deterministic method.
        
        Args:
            cities: List of city names to process
            images_per_city: Number of images to select per city
            selection_method: Selection algorithm ("systematic", "first_n", "evenly_spaced")
        
        Returns:
            List of selected ImageItem objects
        """
        if not self.discovered_items:
            self.discover_all_triplets()
        
        selected_items = []
        
        for city in cities:
            city_items = self.items_by_city.get(city, [])
            
            if not city_items:
                logger.warning(f"No valid items found for city: {city}")
                continue
            
            if len(city_items) < images_per_city:
                logger.warning(f"City {city} has only {len(city_items)} valid images, "
                             f"requested {images_per_city}")
                selected = city_items
            else:
                selected = self._select_by_method(city_items, images_per_city, selection_method)
            
            selected_items.extend(selected)
            logger.info(f"Selected {len(selected)} images from {city}")
        
        logger.info(f"Total selected: {len(selected_items)} images across {len(cities)} cities")
        return selected_items
    
    def _select_by_method(self, items: List[ImageItem], count: int, method: str) -> List[ImageItem]:
        """Apply selection method to choose images"""
        if method == "first_n":
            return items[:count]
        
        elif method == "systematic":
            # Select every nth item to get evenly distributed sample
            if len(items) <= count:
                return items
            
            step = len(items) // count
            selected = []
            
            for i in range(count):
                idx = i * step
                if idx < len(items):
                    selected.append(items[idx])
            
            return selected[:count]
        
        elif method == "evenly_spaced":
            # Select items at evenly spaced intervals
            if len(items) <= count:
                return items
            
            indices = [int(i * (len(items) - 1) / (count - 1)) for i in range(count)]
            return [items[i] for i in indices]
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get dataset statistics"""
        if not self.discovered_items:
            self.discover_all_triplets()
        
        stats = {
            'total_items': len(self.discovered_items),
            'cities': list(self.items_by_city.keys()),
            'items_per_city': {city: len(items) for city, items in self.items_by_city.items()},
            'valid_gps_count': sum(1 for item in self.discovered_items if item.ground_truth_coords),
            'coordinate_ranges': self._calculate_coordinate_ranges()
        }
        
        return stats
    
    def _calculate_coordinate_ranges(self) -> Dict[str, any]:
        """Calculate coordinate ranges for valid GPS data"""
        valid_coords = [item.ground_truth_coords for item in self.discovered_items 
                       if item.ground_truth_coords]
        
        if not valid_coords:
            return {'lat_range': None, 'lon_range': None}
        
        lats = [coord[0] for coord in valid_coords]
        lons = [coord[1] for coord in valid_coords]
        
        return {
            'lat_range': (min(lats), max(lats)),
            'lon_range': (min(lons), max(lons)),
            'center': (sum(lats) / len(lats), sum(lons) / len(lons))
        }
    
    def validate_selection(self, selected_items: List[ImageItem]) -> Dict[str, any]:
        """Validate a selection of items and return validation report"""
        validation_report = {
            'total_items': len(selected_items),
            'all_files_exist': True,
            'all_have_gps': True,
            'missing_files': [],
            'missing_gps': [],
            'cities_represented': set(),
            'coordinate_coverage': None
        }
        
        for item in selected_items:
            validation_report['cities_represented'].add(item.city)
            
            if not item.is_valid():
                validation_report['all_files_exist'] = False
                if not item.image_path.exists():
                    validation_report['missing_files'].append(f"Image: {item.image_path}")
                if not item.annotation_path.exists():
                    validation_report['missing_files'].append(f"Annotation: {item.annotation_path}")
                if not item.gps_path.exists():
                    validation_report['missing_files'].append(f"GPS: {item.gps_path}")
            
            if not item.ground_truth_coords:
                validation_report['all_have_gps'] = False
                validation_report['missing_gps'].append(item.image_id)
        
        # Calculate coordinate coverage
        valid_coords = [item.ground_truth_coords for item in selected_items 
                       if item.ground_truth_coords]
        if valid_coords:
            lats = [coord[0] for coord in valid_coords]
            lons = [coord[1] for coord in valid_coords]
            validation_report['coordinate_coverage'] = {
                'lat_range': (min(lats), max(lats)),
                'lon_range': (min(lons), max(lons)),
                'span_km': self._calculate_geographic_span(valid_coords)
            }
        
        validation_report['cities_represented'] = list(validation_report['cities_represented'])
        return validation_report
    
    def _calculate_geographic_span(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate approximate geographic span in kilometers"""
        if len(coords) < 2:
            return 0.0
        
        lats = [coord[0] for coord in coords]
        lons = [coord[1] for coord in coords]
        
        # Simple approximation using coordinate differences
        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        
        # Approximate conversion to kilometers (rough estimate)
        lat_km = lat_span * 111.0  # 1 degree latitude ≈ 111 km
        lon_km = lon_span * 111.0 * 0.7  # Approximate for central Europe
        
        return (lat_km**2 + lon_km**2)**0.5


if __name__ == "__main__":
    # Quick test/demo
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pipeline_dataset.py <cityscapes_root>")
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test dataset discovery
    dataset = PipelineDataset(sys.argv[1])
    items = dataset.discover_all_triplets()
    
    print("\nDataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test selection
    if stats['cities']:
        selected = dataset.select_images_per_city(
            cities=stats['cities'][:2],  # First 2 cities
            images_per_city=3,
            selection_method="systematic"
        )
        
        print(f"\nSelected {len(selected)} images for testing")
        validation = dataset.validate_selection(selected)
        print("Validation report:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
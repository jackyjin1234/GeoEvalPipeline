#!/usr/bin/python
#
# Distance Calculation Utilities for Geolocation Evaluation
#
# This module provides accurate distance calculations between geographic coordinates
# and related utilities for geolocation error analysis.

from __future__ import print_function, absolute_import, division
import math
from typing import Tuple, List, Dict, Optional

class DistanceCalculator:
    """
    Utility class for calculating distances between geographic coordinates
    and performing geolocation error analysis
    """
    
    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0
    
    def __init__(self):
        pass
    
    def haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate the great-circle distance between two points on Earth using the Haversine formula
        
        Args:
            coord1: (latitude, longitude) of first point in decimal degrees
            coord2: (latitude, longitude) of second point in decimal degrees
        
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        
        c = 2 * math.asin(math.sqrt(a))
        
        # Distance in kilometers
        distance = self.EARTH_RADIUS_KM * c
        
        return distance
    
    def vincenty_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate distance using Vincenty's formulae (more accurate for long distances)
        
        Args:
            coord1: (latitude, longitude) of first point in decimal degrees
            coord2: (latitude, longitude) of second point in decimal degrees
        
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # WGS84 ellipsoid parameters
        a = 6378137.0  # Semi-major axis in meters
        f = 1 / 298.257223563  # Flattening
        b = (1 - f) * a  # Semi-minor axis
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        
        L = lon2_rad - lon1_rad
        U1 = math.atan((1 - f) * math.tan(lat1_rad))
        U2 = math.atan((1 - f) * math.tan(lat2_rad))
        
        sin_U1 = math.sin(U1)
        cos_U1 = math.cos(U1)
        sin_U2 = math.sin(U2)
        cos_U2 = math.cos(U2)
        
        lambda_val = L
        lambda_prev = 2 * math.pi
        
        iteration_limit = 100
        
        while abs(lambda_val - lambda_prev) > 1e-12 and iteration_limit > 0:
            sin_lambda = math.sin(lambda_val)
            cos_lambda = math.cos(lambda_val)
            
            sin_sigma = math.sqrt((cos_U2 * sin_lambda) ** 2 + 
                                (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lambda) ** 2)
            
            if sin_sigma == 0:
                return 0  # Coincident points
            
            cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lambda
            sigma = math.atan2(sin_sigma, cos_sigma)
            
            sin_alpha = cos_U1 * cos_U2 * sin_lambda / sin_sigma
            cos2_alpha = 1 - sin_alpha ** 2
            
            if cos2_alpha == 0:
                cos_2sigma_m = 0  # Equatorial line
            else:
                cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos2_alpha
            
            C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
            
            lambda_prev = lambda_val
            lambda_val = L + (1 - C) * f * sin_alpha * (
                sigma + C * sin_sigma * (cos_2sigma_m + C * cos_sigma * 
                (-1 + 2 * cos_2sigma_m ** 2)))
            
            iteration_limit -= 1
        
        if iteration_limit == 0:
            # Fallback to haversine if Vincenty doesn't converge
            return self.haversine_distance(coord1, coord2)
        
        u2 = cos2_alpha * (a ** 2 - b ** 2) / (b ** 2)
        A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        
        delta_sigma = B * sin_sigma * (cos_2sigma_m + B / 4 * (cos_sigma * 
            (-1 + 2 * cos_2sigma_m ** 2) - B / 6 * cos_2sigma_m * 
            (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos_2sigma_m ** 2)))
        
        s = b * A * (sigma - delta_sigma)
        
        return s / 1000.0  # Convert to kilometers
    
    def bearing(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate the initial bearing from coord1 to coord2
        
        Args:
            coord1: (latitude, longitude) of starting point
            coord2: (latitude, longitude) of destination point
        
        Returns:
            Bearing in degrees (0-360)
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing = math.atan2(y, x)
        bearing_degrees = math.degrees(bearing)
        
        # Normalize to 0-360 degrees
        return (bearing_degrees + 360) % 360
    
    def calculate_error_statistics(self, errors: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive error statistics
        
        Args:
            errors: List of error distances in kilometers
        
        Returns:
            Dictionary with error statistics
        """
        if not errors:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'min': 0.0,
                'max': 0.0,
                'q25': 0.0,
                'q75': 0.0,
                'rmse': 0.0
            }
        
        errors_sorted = sorted(errors)
        n = len(errors)
        
        # Basic statistics
        mean_error = sum(errors) / n
        median_error = errors_sorted[n // 2] if n % 2 == 1 else (errors_sorted[n // 2 - 1] + errors_sorted[n // 2]) / 2
        min_error = min(errors)
        max_error = max(errors)
        
        # Standard deviation
        variance = sum((x - mean_error) ** 2 for x in errors) / n
        std_dev = math.sqrt(variance)
        
        # Quartiles
        q25_idx = n // 4
        q75_idx = 3 * n // 4
        q25 = errors_sorted[q25_idx]
        q75 = errors_sorted[q75_idx]
        
        # Root Mean Square Error
        rmse = math.sqrt(sum(x ** 2 for x in errors) / n)
        
        return {
            'count': n,
            'mean': mean_error,
            'median': median_error,
            'std_dev': std_dev,
            'min': min_error,
            'max': max_error,
            'q25': q25,
            'q75': q75,
            'rmse': rmse
        }
    
    def categorize_error_severity(self, error_km: float) -> str:
        """
        Categorize error severity based on distance
        
        Args:
            error_km: Error distance in kilometers
        
        Returns:
            Error severity category
        """
        if error_km < 1.0:
            return "Excellent"
        elif error_km < 5.0:
            return "Good"
        elif error_km < 25.0:
            return "Fair"
        elif error_km < 100.0:
            return "Poor"
        else:
            return "Very Poor"
    
    def calculate_accuracy_at_distances(self, errors: List[float], 
                                      thresholds: List[float] = None) -> Dict[str, float]:
        """
        Calculate accuracy rates at various distance thresholds
        
        Args:
            errors: List of error distances in kilometers
            thresholds: Distance thresholds to evaluate (default: [1, 5, 25, 100, 1000])
        
        Returns:
            Dictionary mapping threshold to accuracy percentage
        """
        if thresholds is None:
            thresholds = [1, 5, 25, 100, 1000]  # km
        
        if not errors:
            return {f"{t}km": 0.0 for t in thresholds}
        
        total_predictions = len(errors)
        accuracy_rates = {}
        
        for threshold in thresholds:
            correct_predictions = sum(1 for error in errors if error <= threshold)
            accuracy_rate = (correct_predictions / total_predictions) * 100
            accuracy_rates[f"{threshold}km"] = accuracy_rate
        
        return accuracy_rates
    
    def calculate_comparative_statistics(self, original_errors: List[float], 
                                       modified_errors: List[float]) -> Dict[str, any]:
        """
        Calculate comparative statistics between original and modified predictions
        
        Args:
            original_errors: Error distances for original images
            modified_errors: Error distances for feature-removed images
        
        Returns:
            Dictionary with comparative statistics
        """
        if len(original_errors) != len(modified_errors):
            raise ValueError("Error lists must have the same length")
        
        error_increases = [mod - orig for orig, mod in zip(original_errors, modified_errors)]
        
        original_stats = self.calculate_error_statistics(original_errors)
        modified_stats = self.calculate_error_statistics(modified_errors)
        increase_stats = self.calculate_error_statistics(error_increases)
        
        # Calculate improvement/degradation rates
        improved_count = sum(1 for diff in error_increases if diff < 0)
        degraded_count = sum(1 for diff in error_increases if diff > 0)
        unchanged_count = len(error_increases) - improved_count - degraded_count
        
        total = len(error_increases)
        
        return {
            'original_stats': original_stats,
            'modified_stats': modified_stats,
            'error_increase_stats': increase_stats,
            'improvement_rate': (improved_count / total) * 100 if total > 0 else 0,
            'degradation_rate': (degraded_count / total) * 100 if total > 0 else 0,
            'unchanged_rate': (unchanged_count / total) * 100 if total > 0 else 0,
            'median_error_increase': increase_stats['median'],
            'mean_error_increase': increase_stats['mean']
        }
    
    def validate_coordinates(self, coord: Tuple[float, float]) -> bool:
        """
        Validate that coordinates are within valid ranges
        
        Args:
            coord: (latitude, longitude) tuple
        
        Returns:
            True if coordinates are valid, False otherwise
        """
        lat, lon = coord
        return -90 <= lat <= 90 and -180 <= lon <= 180


# Utility functions for testing and validation
def test_distance_calculations():
    """Test distance calculations with known coordinates"""
    calc = DistanceCalculator()
    
    # Test cases with known distances
    test_cases = [
        # New York to Los Angeles (approximately 3944 km)
        ((40.7128, -74.0060), (34.0522, -118.2437), 3944),
        # London to Paris (approximately 344 km)
        ((51.5074, -0.1278), (48.8566, 2.3522), 344),
        # Same point (0 km)
        ((52.5200, 13.4050), (52.5200, 13.4050), 0)
    ]
    
    print("Testing distance calculations:")
    for coord1, coord2, expected in test_cases:
        haversine_dist = calc.haversine_distance(coord1, coord2)
        vincenty_dist = calc.vincenty_distance(coord1, coord2)
        
        print(f"{coord1} to {coord2}:")
        print(f"  Expected: ~{expected} km")
        print(f"  Haversine: {haversine_dist:.2f} km")
        print(f"  Vincenty: {vincenty_dist:.2f} km")
        print()


if __name__ == "__main__":
    test_distance_calculations()
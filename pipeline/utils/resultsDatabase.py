#!/usr/bin/python
#
# SQLite Results Database for Geolocation Feature Evaluation
#
# This module manages structured storage of evaluation results with efficient
# querying and analysis capabilities.

from __future__ import print_function, absolute_import, division
import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

class ResultsDatabase:
    """
    SQLite database manager for geolocation evaluation results
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_dir = os.path.dirname(db_path)
        
        # Create directory if it doesn't exist
        if self.db_dir:
            os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    feature_removed TEXT NOT NULL,
                    original_error_km REAL,
                    modified_error_km REAL,
                    error_increase_km REAL,
                    predicted_lat REAL,
                    predicted_lon REAL,
                    ground_truth_lat REAL,
                    ground_truth_lon REAL,
                    confidence TEXT,
                    region TEXT,
                    model_used TEXT,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    gpt_reasoning TEXT,
                    critical_features TEXT,
                    raw_response TEXT,
                    api_usage TEXT
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_image_id ON evaluation_results(image_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_feature_removed ON evaluation_results(feature_removed)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_error_increase ON evaluation_results(error_increase_km)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_confidence ON evaluation_results(confidence)
            ''')
            
            # Summary statistics table for quick access
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS summary_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_removed TEXT NOT NULL,
                    total_images INTEGER,
                    successful_predictions INTEGER,
                    avg_original_error REAL,
                    avg_modified_error REAL,
                    avg_error_increase REAL,
                    median_error_increase REAL,
                    max_error_increase REAL,
                    min_error_increase REAL,
                    std_dev_error_increase REAL,
                    success_rate REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(feature_removed)
                )
            ''')
            
            # Metadata table for evaluation run information
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE,
                    input_directory TEXT,
                    model_used TEXT,
                    features_tested TEXT,
                    total_images INTEGER,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_processing_time REAL,
                    total_api_cost REAL,
                    notes TEXT
                )
            ''')
            
            conn.commit()
    
    def insert_result(self, result: Dict[str, Any]) -> int:
        """
        Insert a single evaluation result
        
        Args:
            result: Dictionary containing evaluation result data
        
        Returns:
            Row ID of inserted result
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Extract coordinate data from GPT response
            predicted_lat = predicted_lon = None
            ground_truth_lat = ground_truth_lon = None
            confidence = region = model_used = None
            gpt_reasoning = critical_features = raw_response = None
            api_usage = None
            
            if result.get('gpt_response'):
                gpt_resp = result['gpt_response']
                if 'coordinates' in gpt_resp:
                    predicted_lat, predicted_lon = gpt_resp['coordinates']
                confidence = gpt_resp.get('confidence')
                region = gpt_resp.get('region')
                model_used = gpt_resp.get('model_used')
                gpt_reasoning = gpt_resp.get('reasoning')
                critical_features = gpt_resp.get('critical_features')
                raw_response = gpt_resp.get('raw_response')
                
                if 'api_usage' in gpt_resp:
                    api_usage = json.dumps(gpt_resp['api_usage'])
            
            cursor.execute('''
                INSERT INTO evaluation_results (
                    image_id, feature_removed, original_error_km, modified_error_km,
                    error_increase_km, predicted_lat, predicted_lon, ground_truth_lat,
                    ground_truth_lon, confidence, region, model_used, processing_time,
                    gpt_reasoning, critical_features, raw_response, api_usage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['image_id'],
                result['feature_removed'],
                result.get('original_error_km'),
                result.get('modified_error_km'),
                result.get('error_increase_km'),
                predicted_lat,
                predicted_lon,
                ground_truth_lat,  # TODO: Extract from ground truth when available
                ground_truth_lon,  # TODO: Extract from ground truth when available
                confidence,
                region,
                model_used,
                result.get('processing_time', 0),
                gpt_reasoning,
                critical_features,
                raw_response,
                api_usage
            ))
            
            row_id = cursor.lastrowid
            conn.commit()
            
            # Update summary statistics
            self._update_summary_statistics(result['feature_removed'])
            
            return row_id
    
    def batch_insert_results(self, results: List[Dict[str, Any]]) -> List[int]:
        """
        Insert multiple results efficiently
        
        Args:
            results: List of result dictionaries
        
        Returns:
            List of row IDs
        """
        row_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for result in results:
                # Use the same logic as insert_result but in a transaction
                predicted_lat = predicted_lon = None
                ground_truth_lat = ground_truth_lon = None
                confidence = region = model_used = None
                gpt_reasoning = critical_features = raw_response = None
                api_usage = None
                
                if result.get('gpt_response'):
                    gpt_resp = result['gpt_response']
                    if 'coordinates' in gpt_resp:
                        predicted_lat, predicted_lon = gpt_resp['coordinates']
                    confidence = gpt_resp.get('confidence')
                    region = gpt_resp.get('region')
                    model_used = gpt_resp.get('model_used')
                    gpt_reasoning = gpt_resp.get('reasoning')
                    critical_features = gpt_resp.get('critical_features')
                    raw_response = gpt_resp.get('raw_response')
                    
                    if 'api_usage' in gpt_resp:
                        api_usage = json.dumps(gpt_resp['api_usage'])
                
                cursor.execute('''
                    INSERT INTO evaluation_results (
                        image_id, feature_removed, original_error_km, modified_error_km,
                        error_increase_km, predicted_lat, predicted_lon, ground_truth_lat,
                        ground_truth_lon, confidence, region, model_used, processing_time,
                        gpt_reasoning, critical_features, raw_response, api_usage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result['image_id'],
                    result['feature_removed'],
                    result.get('original_error_km'),
                    result.get('modified_error_km'),
                    result.get('error_increase_km'),
                    predicted_lat,
                    predicted_lon,
                    ground_truth_lat,
                    ground_truth_lon,
                    confidence,
                    region,
                    model_used,
                    result.get('processing_time', 0),
                    gpt_reasoning,
                    critical_features,
                    raw_response,
                    api_usage
                ))
                
                row_ids.append(cursor.lastrowid)
            
            conn.commit()
        
        # Update summary statistics for all features
        features = set(result['feature_removed'] for result in results)
        for feature in features:
            self._update_summary_statistics(feature)
        
        return row_ids
    
    def _update_summary_statistics(self, feature_removed: str):
        """Update summary statistics for a specific feature"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_images,
                    COUNT(modified_error_km) as successful_predictions,
                    AVG(original_error_km) as avg_original_error,
                    AVG(modified_error_km) as avg_modified_error,
                    AVG(error_increase_km) as avg_error_increase,
                    MAX(error_increase_km) as max_error_increase,
                    MIN(error_increase_km) as min_error_increase
                FROM evaluation_results 
                WHERE feature_removed = ?
            ''', (feature_removed,))
            
            stats = cursor.fetchone()
            
            if stats and stats[0] > 0:  # If we have data
                total_images, successful_predictions, avg_original, avg_modified, avg_increase, max_increase, min_increase = stats
                
                success_rate = (successful_predictions / total_images * 100) if total_images > 0 else 0
                
                # Calculate median and standard deviation
                cursor.execute('''
                    SELECT error_increase_km FROM evaluation_results 
                    WHERE feature_removed = ? AND error_increase_km IS NOT NULL
                    ORDER BY error_increase_km
                ''', (feature_removed,))
                
                error_increases = [row[0] for row in cursor.fetchall()]
                
                median_increase = None
                std_dev_increase = None
                
                if error_increases:
                    n = len(error_increases)
                    median_increase = error_increases[n // 2] if n % 2 == 1 else (error_increases[n // 2 - 1] + error_increases[n // 2]) / 2
                    
                    if avg_increase is not None and n > 1:
                        variance = sum((x - avg_increase) ** 2 for x in error_increases) / n
                        std_dev_increase = variance ** 0.5
                
                # Upsert summary statistics
                cursor.execute('''
                    INSERT OR REPLACE INTO summary_statistics (
                        feature_removed, total_images, successful_predictions,
                        avg_original_error, avg_modified_error, avg_error_increase,
                        median_error_increase, max_error_increase, min_error_increase,
                        std_dev_error_increase, success_rate, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    feature_removed, total_images, successful_predictions,
                    avg_original, avg_modified, avg_increase,
                    median_increase, max_increase, min_increase,
                    std_dev_increase, success_rate
                ))
                
                conn.commit()
    
    def get_results_by_feature(self, feature_removed: str) -> List[Dict[str, Any]]:
        """Get all results for a specific feature"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM evaluation_results 
                WHERE feature_removed = ?
                ORDER BY image_id
            ''', (feature_removed,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all features"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM summary_statistics 
                ORDER BY feature_removed
            ''')
            
            summary = {}
            for row in cursor.fetchall():
                feature = row['feature_removed']
                summary[feature] = {
                    'total_images': row['total_images'],
                    'successful_predictions': row['successful_predictions'],
                    'avg_error': row['avg_original_error'] if feature == 'none' else row['avg_modified_error'],
                    'avg_error_increase': row['avg_error_increase'],
                    'median_error': row['median_error_increase'],
                    'median_error_increase': row['median_error_increase'],
                    'max_error_increase': row['max_error_increase'],
                    'min_error_increase': row['min_error_increase'],
                    'std_dev_error_increase': row['std_dev_error_increase'],
                    'success_rate': row['success_rate'],
                    'impact_severity': self._categorize_impact(row['avg_error_increase'])
                }
            
            return summary
    
    def _categorize_impact(self, avg_error_increase: Optional[float]) -> str:
        """Categorize the impact severity based on average error increase"""
        if avg_error_increase is None:
            return "Unknown"
        elif avg_error_increase < 1.0:
            return "Minimal"
        elif avg_error_increase < 5.0:
            return "Low"
        elif avg_error_increase < 25.0:
            return "Moderate"
        elif avg_error_increase < 100.0:
            return "High"
        else:
            return "Critical"
    
    def export_to_csv(self, output_path: str, feature_removed: Optional[str] = None):
        """Export results to CSV format"""
        import csv
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if feature_removed:
                cursor.execute('''
                    SELECT * FROM evaluation_results 
                    WHERE feature_removed = ?
                    ORDER BY image_id
                ''', (feature_removed,))
            else:
                cursor.execute('''
                    SELECT * FROM evaluation_results 
                    ORDER BY feature_removed, image_id
                ''')
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)
                writer.writerows(cursor.fetchall())
    
    def export_to_json(self, output_path: str, feature_removed: Optional[str] = None):
        """Export results to JSON format"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if feature_removed:
                cursor.execute('''
                    SELECT * FROM evaluation_results 
                    WHERE feature_removed = ?
                    ORDER BY image_id
                ''', (feature_removed,))
            else:
                cursor.execute('''
                    SELECT * FROM evaluation_results 
                    ORDER BY feature_removed, image_id
                ''')
            
            results = [dict(row) for row in cursor.fetchall()]
            
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(results, jsonfile, indent=2, default=str)
    
    def get_comparative_analysis(self) -> Dict[str, Any]:
        """Get comparative analysis across all features"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get feature ranking by error increase
            cursor.execute('''
                SELECT feature_removed, avg_error_increase 
                FROM summary_statistics 
                WHERE feature_removed != 'none' AND avg_error_increase IS NOT NULL
                ORDER BY avg_error_increase DESC
            ''')
            
            feature_ranking = cursor.fetchall()
            
            # Get overall statistics
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT image_id) as total_unique_images,
                    COUNT(*) as total_evaluations,
                    AVG(processing_time) as avg_processing_time,
                    SUM(processing_time) as total_processing_time
                FROM evaluation_results
            ''')
            
            overall_stats = cursor.fetchone()
            
            return {
                'feature_importance_ranking': feature_ranking,
                'total_unique_images': overall_stats[0],
                'total_evaluations': overall_stats[1],
                'avg_processing_time': overall_stats[2],
                'total_processing_time': overall_stats[3]
            }
    
    def close(self):
        """Close database connection (if needed for cleanup)"""
        # SQLite connections are automatically closed when using context managers
        pass


# Utility functions
def migrate_database(db_path: str):
    """Migrate database schema to latest version"""
    # This function can be expanded to handle schema migrations
    # For now, it just ensures the database is initialized
    db = ResultsDatabase(db_path)
    print(f"Database initialized/migrated: {db_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python resultsDatabase.py <database_path>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    # Test database initialization
    db = ResultsDatabase(db_path)
    print(f"Database initialized: {db_path}")
    
    # Show current statistics if database has data
    summary = db.get_summary_statistics()
    if summary:
        print("\nCurrent Summary Statistics:")
        for feature, stats in summary.items():
            print(f"{feature}: {stats['total_images']} images, {stats['success_rate']:.1f}% success rate")
    else:
        print("Database is empty.")
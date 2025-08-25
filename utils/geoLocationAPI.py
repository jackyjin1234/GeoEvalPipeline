#!/usr/bin/python
#
# OpenAI GPT Geolocation API Integration
#
# This module handles communication with OpenAI's vision models for geolocation tasks.
# It includes optimized prompts, response parsing, caching, and error handling.

from __future__ import print_function, absolute_import, division
import os, json, time, base64, hashlib
import asyncio, aiohttp
from pathlib import Path
import re
from typing import Dict, Optional, Tuple, Any
from dotenv import load_dotenv
import openai

GEOLOCATION_FUNCTION = {
    "name": "get_geolocation",
    "description": "Analyze the image and return a geolocation estimate with reasoning and key features as a JSON object.",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Step-by-step analysis of visual clues."
            },
            "location_estimate": {
                "type": "object",
                "properties": {
                    "latitude":   {"type": "number", "description": "Latitude in decimal degrees"},
                    "longitude":  {"type": "number", "description": "Longitude in decimal degrees"},
                    "confidence": {
                        "type": "string",
                        "enum": ["High","Medium","Low"],
                        "description": "Model's confidence level"
                    },
                    "region": {"type": "string", "description": "Country, State/Province, City if known"}
                },
                "required": ["latitude", "longitude", "confidence", "region"]
            },
            "critical_features": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
                "description": "Top 3-5 visual elements that drove the estimate"
            }
        },
        "required": ["reasoning", "location_estimate", "critical_features"]
    }
}

class GeoLocationAPI:
    """
    Handles OpenAI GPT vision API calls for geolocation tasks
    """
    
    def __init__(self, model="gpt-4o", concurrent_requests=5, cache_responses=False):
        load_dotenv()
        self.model = model
        self.concurrent_requests = concurrent_requests
        self.cache_responses = cache_responses
        self.cache_dir = Path.home() / '.cache' / 'cityscapes_geolocation'
        
        if cache_responses:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second max
        
        # Optimized prompts for geolocation
        self.base_prompt = self._create_geolocation_prompt()
    

    
    def _create_geolocation_prompt(self):
        return """
    You are an expert geographer and urban analyst. Analyze this street-view image and determine its most likely location.

    **You must return a single JSON object matching the ‘get_geolocation’ schema exactly. Do not include any extra explanation.**
    """


    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for image"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return hashlib.md5(image_data).hexdigest()
    
    def _load_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Load cached response if available"""
        if not self.cache_responses:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None
    
    def _save_cached_response(self, cache_key: str, response: Dict):
        """Save response to cache"""
        if not self.cache_responses:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f, indent=2)
        except IOError:
            pass  # Cache write failure is not critical
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image as base64 string"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _parse_gpt_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse GPT response to extract structured location data
        """
        try:
            # Extract coordinates using regex
            lat_match = re.search(r'Latitude:\s*([-+]?\d*\.?\d+)', response_text)
            lon_match = re.search(r'Longitude:\s*([-+]?\d*\.?\d+)', response_text)
            
            if not lat_match or not lon_match:
                return None
            
            latitude = float(lat_match.group(1))
            longitude = float(lon_match.group(1))
            
            # Extract confidence
            confidence_match = re.search(r'Confidence:\s*([^\n]+)', response_text)
            confidence = confidence_match.group(1).strip() if confidence_match else "Unknown"
            
            # Extract region
            region_match = re.search(r'Region:\s*([^\n]+)', response_text)
            region = region_match.group(1).strip() if region_match else "Unknown"
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*\n(.*?)\n\nLOCATION_ESTIMATE:', response_text, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # Extract critical features
            features_match = re.search(r'CRITICAL_FEATURES:\s*\n(.*?)(?:\n\n|\Z)', response_text, re.DOTALL)
            critical_features = features_match.group(1).strip() if features_match else ""
            
            return {
                'coordinates': (latitude, longitude),
                'confidence': confidence,
                'region': region,
                'reasoning': reasoning,
                'critical_features': critical_features,
                'raw_response': response_text
            }
            
        except (ValueError, AttributeError) as e:
            print(f"Error parsing GPT response: {e}")
            return None
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def get_geolocation_prediction(self, image_path: str) -> Optional[Dict]:
        """
        Get geolocation prediction for a single image
        
        Returns:
            Dict with prediction data or None if failed
        """
        client = openai.AsyncOpenAI(api_key = os.getenv('OPENAI_API_KEY'))
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(image_path)
        cached_response = self._load_cached_response(cache_key)
        if cached_response:
            cached_response['processing_time'] = 0  # Cached response
            return cached_response

        async with self.semaphore:
            await self._rate_limit()
            try:
                # Encode image
                base64_image = self._encode_image_base64(image_path)

                # Make API request using modern OpenAI library
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.base_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    functions=[GEOLOCATION_FUNCTION],
                    function_call={"name": "get_geolocation"},
                    max_tokens=1000,
                    temperature=0.1,
                    timeout=30
                )

                # Extract and parse the response
                msg = response.choices[0].message
                if msg.function_call and msg.function_call.name == "get_geolocation":
                    parsed = json.loads(msg.function_call.arguments)

                    # Add your metadata:
                    parsed["processing_time"] = time.time() - start_time
                    parsed["model_used"]       = self.model

                if not parsed:
                    print(f"Failed to parse response for {image_path}")
                    return None

                # Cache and return
                self._save_cached_response(cache_key, parsed)
                return parsed

            except openai.APITimeoutError:
                print(f"Timeout for image: {image_path}")
                return None

            except openai.APIError as e:
                print(f"API error for {image_path}: {e}")
                return None

            except openai.OpenAIError as e:
                # Catches other OpenAI-specific errors (RateLimitError, AuthenticationError, etc.)
                print(f"OpenAI error for {image_path}: {e}")
                return None

            except Exception as e:
                # Any other unexpected exception
                print(f"Unexpected error for {image_path}: {e}")
                return None

    async def batch_predict(self, image_paths: list) -> Dict[str, Optional[Dict]]:
        """
        Process multiple images concurrently
        
        Returns:
            Dict mapping image paths to prediction results
        """
        tasks = [self.get_geolocation_prediction(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            path: result if not isinstance(result, Exception) else None
            for path, result in zip(image_paths, results)
        }
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        if not self.cache_responses or not self.cache_dir.exists():
            return {'cached_responses': 0}
        
        cache_files = list(self.cache_dir.glob('*.json'))
        return {
            'cached_responses': len(cache_files),
            'cache_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        }
    
    def clear_cache(self):
        """Clear response cache"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob('*.json'):
                cache_file.unlink()


# Utility functions for standalone usage
async def test_single_image(image_path: str, model: str = "gpt-4o"):
    """Test geolocation prediction on a single image"""
    api = GeoLocationAPI(model=model, cache_responses=True)
    result = await api.get_geolocation_prediction(image_path)
    
    if result:
        print(f"Image: {image_path}")
        print(f"Predicted coordinates: {result['location_estimate']['latitude']}, {result['location_estimate']['longitude']}")
        print(f"Confidence: {result['location_estimate']['confidence']}")
        print(f"Region: {result['location_estimate']['region']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Critical features: {result['critical_features']}")
    else:
        print(f"Failed to get prediction for {image_path}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python geoLocationAPI.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    
    # Test the API with a single image
    asyncio.run(test_single_image(image_path))
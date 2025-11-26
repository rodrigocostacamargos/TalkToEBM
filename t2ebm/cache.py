"""
Cache module for storing simplified graph text representations and LLM responses to avoid reprocessing.
"""

import hashlib
import json
import os
import time
from typing import Optional, Dict, Any
import pickle

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)


class GraphCache:
    """Cache for storing simplified graph text representations."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, ebm, feature_index: int, **kwargs) -> str:
        """Generate a unique cache key for EBM + feature + parameters."""
        # Create a hash from EBM metadata and parameters
        hash_data = {
            "feature_names": list(ebm.feature_names_in_),
            "feature_types": list(ebm.feature_types_in_),
            "feature_index": feature_index,
            "bins": [list(b) for b in ebm.bins_],
            "kwargs": kwargs
        }
        
        # Convert to JSON string and hash
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def get_cached_graph_text(self, ebm, feature_index: int, **kwargs) -> Optional[str]:
        """Get cached graph text if it exists."""
        cache_key = self._generate_cache_key(ebm, feature_index, **kwargs)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                return cache_data.get('graph_text')
            except (json.JSONDecodeError, KeyError):
                return None
        return None
    
    def set_cached_graph_text(self, ebm, feature_index: int, graph_text: str, **kwargs):
        """Cache graph text for future use."""
        cache_key = self._generate_cache_key(ebm, feature_index, **kwargs)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        cache_data = {
            'graph_text': graph_text,
            'feature_name': ebm.feature_names_in_[feature_index],
            'feature_index': feature_index,
            'kwargs': kwargs
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def clear_cache(self):
        """Clear all cached graph data."""
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))


class LLMResponseCache:
    """Cache for storing LLM responses to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = ".cache/llm_responses", ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, model: str, messages: list, **kwargs) -> str:
        """Generate a unique cache key for model + messages + parameters."""
        hash_data = {
            "model": model,
            "messages": messages,
            "kwargs": kwargs
        }
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def get_cached_response(self, model: str, messages: list, **kwargs) -> Optional[str]:
        """Get cached LLM response if it exists and is not expired."""
        cache_key = self._generate_cache_key(model, messages, **kwargs)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check TTL
                cached_time = cache_data.get('timestamp', 0)
                if time.time() - cached_time < self.ttl_seconds:
                    print(f"[CACHE HIT] Using cached LLM response")
                    return cache_data.get('response')
                else:
                    # Cache expired
                    os.remove(cache_file)
            except (json.JSONDecodeError, KeyError):
                return None
        return None
    
    def set_cached_response(self, model: str, messages: list, response: str, **kwargs):
        """Cache LLM response for future use."""
        cache_key = self._generate_cache_key(model, messages, **kwargs)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        cache_data = {
            'model': model,
            'response': response,
            'timestamp': time.time(),
            'kwargs': kwargs
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"[CACHE SET] Cached LLM response")
    
    def clear_cache(self):
        """Clear all cached LLM responses."""
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not os.path.exists(self.cache_dir):
            return {"count": 0, "size_mb": 0}
        
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) 
            for f in files
        )
        
        return {
            "count": len(files),
            "size_mb": round(total_size / (1024 * 1024), 2)
        }


# Global cache instances
_graph_cache = GraphCache()
_llm_cache = LLMResponseCache()


def get_graph_cache() -> GraphCache:
    """Get the global graph cache instance."""
    return _graph_cache


def get_llm_cache() -> LLMResponseCache:
    """Get the global LLM response cache instance."""
    return _llm_cache


def clear_graph_cache():
    """Clear the global graph cache."""
    _graph_cache.clear_cache()


def clear_llm_cache():
    """Clear the global LLM response cache."""
    _llm_cache.clear_cache()


def clear_all_caches():
    """Clear all caches."""
    clear_graph_cache()
    clear_llm_cache()

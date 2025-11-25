"""
Cache module for storing simplified graph text representations to avoid reprocessing.
"""

import hashlib
import json
import os
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


# Global cache instance
_graph_cache = GraphCache()


def get_graph_cache() -> GraphCache:
    """Get the global graph cache instance."""
    return _graph_cache


def clear_graph_cache():
    """Clear the global graph cache."""
    _graph_cache.clear_cache()

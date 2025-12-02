"""
Cache Manager for Persistent Storage of Processed Protein Graphs

This module provides thread-safe persistent caching functionality for protein graphs,
maintaining subdirectory structure and supporting efficient lookups.
"""

import os
import torch
import threading
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict
import time


class CacheManager:
    """Persistent cache manager for protein graph data with multiprocessing support."""
    
    def __init__(self, cache_path: str, max_size: int = 100000):
        """
        Initialize cache manager.
        
        Args:
            cache_path: Directory to store cached files
            max_size: Maximum number of items to keep in memory cache
        """
        self.cache_path = Path(cache_path)
        self.cache_root = self.cache_path  # Add missing cache_root attribute
        self.max_size = max_size
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Memory cache with LRU eviction
        self._memory_cache = OrderedDict()
        self._lock = threading.Lock()
        
        # Stats
        self._hits = 0
        self._misses = 0
        
        print(f"CacheManager: Initialized with cache_path={cache_path}, max_size={max_size}")
    
    def __getstate__(self):
        """Custom pickling - exclude lock from state."""
        state = self.__dict__.copy()
        # Remove unpickleable lock
        del state['_lock']
        return state
    
    def __setstate__(self, state):
        """Custom unpickling - restore state and create new lock."""
        self.__dict__.update(state)
        # Create new lock in worker process
        self._lock = threading.Lock()
    
    def _get_cache_path(self, pdb_file_path: str, maintain_structure: bool = False) -> Path:
        """
        Get cache file path for a given PDB file.
        
        Args:
            pdb_file_path: Original PDB file path
            maintain_structure: Whether to maintain subdirectory structure
            
        Returns:
            Path to cache file
        """
        pdb_path = Path(pdb_file_path)
        
        if maintain_structure:
            # Maintain relative subdirectory structure
            # Extract relative path from dataset root
            try:
                # Try to find 'dataset' in the path
                path_parts = pdb_path.parts
                if 'dataset' in path_parts:
                    dataset_idx = path_parts.index('dataset')
                    # Get everything after 'dataset' except the filename
                    rel_parts = path_parts[dataset_idx + 1:-1]
                    cache_subdir = self.cache_root / Path(*rel_parts)
                else:
                    # Fallback: use parent directory name
                    cache_subdir = self.cache_root / pdb_path.parent.name
            except (ValueError, IndexError):
                # Fallback: use flattened structure
                cache_subdir = self.cache_root
        else:
            # Flattened structure
            cache_subdir = self.cache_root
        
        # Create subdirectory if it doesn't exist
        cache_subdir.mkdir(parents=True, exist_ok=True)

        pdb_ccd_id = '_'.join(pdb_path.stem.split('_')[:-1])

        # Generate cache filename: pyg_graph_{original_stem}.pt
        cache_filename = f"pyg_graph_{pdb_ccd_id}_esmc_600m.pt"
        return cache_subdir / cache_filename
    
    def _get_file_hash(self, file_path: str) -> Optional[str]:
        """
        Get SHA256 hash of file for validation (optional, since PDB files are immutable).
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash or None if file doesn't exist
        """
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except (IOError, OSError):
            return None
    
    def exists(self, pdb_file_path: str) -> bool:
        """
        Check if cached graph exists for given PDB file.
        
        Args:
            pdb_file_path: Original PDB file path
            
        Returns:
            True if cache exists, False otherwise
        """
        cache_path = self._get_cache_path(pdb_file_path)
        return cache_path.exists()
    
    def load(self, pdb_file_path: str):
        """
        Load cached graph data.
        
        Args:
            pdb_file_path: Original PDB file path
            
        Returns:
            Graph data (Data object or dictionary) or None if not found/invalid
        """
        with self._lock:
            try:
                cache_path = self._get_cache_path(pdb_file_path)
                
                if not cache_path.exists():
                    return None
                
                # Load cached data
                graph_data = torch.load(cache_path, map_location='cpu')
                
                # Validate cached data structure
                from torch_geometric.data import Data
                
                if isinstance(graph_data, Data):
                    # Validate Data object
                    if graph_data.x is None or graph_data.edge_index is None or graph_data.pos is None:
                        print(f"Cache Manager: Invalid cached Data object for {pdb_file_path}, removing cache file")
                        cache_path.unlink()  # Remove invalid cache file
                        return None
                else:
                    # Legacy dictionary format validation
                    required_keys = ['node_features', 'edge_index', 'node_pos']
                    if not all(key in graph_data for key in required_keys):
                        print(f"Cache Manager: Invalid cached data for {pdb_file_path}, removing cache file")
                        cache_path.unlink()  # Remove invalid cache file
                        return None
                    
                    # Ensure edge_attr exists for dictionary format
                    if 'edge_attr' not in graph_data:
                        num_edges = graph_data['edge_index'].shape[1]
                        graph_data['edge_attr'] = torch.ones((num_edges, 1))
                
                print(f"Cache Manager: Loaded cached graph for {pdb_file_path}")
                return graph_data
                
            except Exception as e:
                print(f"Cache Manager: Error loading cache for {pdb_file_path}: {e}")
                # Try to remove corrupted cache file
                try:
                    cache_path = self._get_cache_path(pdb_file_path)
                    if cache_path.exists():
                        cache_path.unlink()
                        print(f"Cache Manager: Removed corrupted cache file: {cache_path}")
                except:
                    pass
                return None
    
    def save(self, pdb_file_path: str, graph_data: Dict[str, torch.Tensor]) -> bool:
        """
        Save graph data to cache.
        
        Args:
            pdb_file_path: Original PDB file path
            graph_data: Graph data dictionary to cache
            
        Returns:
            True if saved successfully, False otherwise
        """
        with self._lock:
            try:
                cache_path = self._get_cache_path(pdb_file_path)
                
                # Validate data before saving
                from torch_geometric.data import Data
                
                # Check if it's a PyTorch Geometric Data object
                if isinstance(graph_data, Data):
                    # Validate Data object has required attributes
                    if graph_data.x is None or graph_data.edge_index is None or graph_data.pos is None:
                        print(f"Cache Manager: Invalid Data object, missing required attributes, cannot cache {pdb_file_path}")
                        return False
                else:
                    # Legacy dictionary format validation
                    required_keys = ['node_features', 'edge_index', 'node_pos']
                    if not all(key in graph_data for key in required_keys):
                        print(f"Cache Manager: Invalid graph data, cannot cache {pdb_file_path}")
                        return False
                
                # Save to temporary file first, then rename (atomic operation)
                temp_path = cache_path.with_suffix('.tmp')
                torch.save(graph_data, temp_path)
                temp_path.rename(cache_path)
                
                print(f"Cache Manager: Cached graph for {pdb_file_path} -> {cache_path}")
                return True
                
            except Exception as e:
                print(f"Cache Manager: Error saving cache for {pdb_file_path}: {e}")
                # Clean up temporary file if it exists
                try:
                    temp_path = self._get_cache_path(pdb_file_path).with_suffix('.tmp')
                    if temp_path.exists():
                        temp_path.unlink()
                except:
                    pass
                return False
    
    def remove(self, pdb_file_path: str) -> bool:
        """
        Remove cached graph data.
        
        Args:
            pdb_file_path: Original PDB file path
            
        Returns:
            True if removed successfully, False otherwise
        """
        with self._lock:
            try:
                cache_path = self._get_cache_path(pdb_file_path)
                if cache_path.exists():
                    cache_path.unlink()
                    print(f"Cache Manager: Removed cache for {pdb_file_path}")
                return True
            except Exception as e:
                print(f"Cache Manager: Error removing cache for {pdb_file_path}: {e}")
                return False
    
    def clear_all(self) -> int:
        """
        Clear all cached data.
        
        Returns:
            Number of files removed
        """
        with self._lock:
            removed_count = 0
            try:
                for cache_file in self.cache_root.rglob("*.pt"):
                    try:
                        cache_file.unlink()
                        removed_count += 1
                    except:
                        pass
                print(f"Cache Manager: Cleared {removed_count} cache files")
            except Exception as e:
                print(f"Cache Manager: Error clearing cache: {e}")
            return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_root.rglob("*.pt"))
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())
            
            return {
                'cache_root': str(self.cache_root),
                'num_cached_files': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'subdirectories': len(list(self.cache_root.rglob("*/")))
            }
        except Exception as e:
            print(f"Cache Manager: Error getting stats: {e}")
            return {
                'cache_root': str(self.cache_root),
                'num_cached_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'subdirectories': 0,
                'error': str(e)
            }
    
    def validate_cache(self) -> Dict[str, int]:
        """
        Validate all cached files and remove corrupted ones.
        
        Returns:
            Dictionary with validation results
        """
        with self._lock:
            stats = {'valid': 0, 'invalid': 0, 'removed': 0}
            
            try:
                for cache_file in self.cache_root.rglob("*.pt"):
                    try:
                        # Try to load the file
                        graph_data = torch.load(cache_file, map_location='cpu')
                        
                        # Validate structure
                        required_keys = ['node_features', 'edge_index', 'node_pos']
                        if all(key in graph_data for key in required_keys):
                            stats['valid'] += 1
                        else:
                            stats['invalid'] += 1
                            cache_file.unlink()
                            stats['removed'] += 1
                            print(f"Cache Manager: Removed invalid cache file: {cache_file}")
                            
                    except Exception:
                        stats['invalid'] += 1
                        try:
                            cache_file.unlink()
                            stats['removed'] += 1
                            print(f"Cache Manager: Removed corrupted cache file: {cache_file}")
                        except:
                            pass
                            
            except Exception as e:
                print(f"Cache Manager: Error during validation: {e}")
            
            print(f"Cache Manager: Validation complete - Valid: {stats['valid']}, Invalid: {stats['invalid']}, Removed: {stats['removed']}")
            return stats

import torch
from torch.utils.data import Dataset
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from tools.subgraph_sampler import ProteinSubgraphSampler
from .pdb_processor import PDBProcessor
from .cache_manager import CacheManager
import random
import threading
from collections import OrderedDict

class ProteinDataset(Dataset):
    def __init__(self, train=True, data_path=None, pdb_path=None, cache_path=None, transform=None, **kwargs):
        self.train = train
        self.data_path = data_path
        self.pdb_path = pdb_path or "dataset/rcsb/human"  # Default PDB path
        self.cache_path = cache_path or "dataset/protein_cache"  # Default cache path
        self.transform = transform
        
        # New parameters for on-the-fly processing
        self.enable_pdb_processing = kwargs.get('enable_pdb_processing', True)
        self.force_pdb_processing = kwargs.get('force_pdb_processing', False)  # Force PDB processing even if .pt files exist
        
        # Subgraph sampling parameters
        self.min_nodes = kwargs.get('min_nodes', 10)
        self.max_nodes = kwargs.get('max_nodes', 100)

        # Initialize components
        self.subgraph_sampler = ProteinSubgraphSampler(
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes
        )
        
        # Initialize PDB processor and cache manager
        self.pdb_processor = None  # Lazy initialization
        self.cache_manager = CacheManager(self.cache_path)
        
        # Determine data source and index files
        self._determine_data_source()
        self.file_paths = self._get_file_paths()
        self.protein_ids = self._generate_protein_ids()
        
        # Split the dataset
        self.seed = kwargs.get('seed', 42)
        self._create_data_splits()

        # Improved memory management for multiprocessing
        self.cache_size = kwargs.get('cache_size', 512)  # Reduced default for multiprocessing
        self.enable_cache = kwargs.get('enable_cache', True)
        self.preload_cache = kwargs.get('preload_cache', False)
        
        # Thread-safe cache using OrderedDict (LRU)
        self._cache_lock = threading.Lock()
        self._cache = OrderedDict()
        
        # Worker-specific settings
        self._worker_id = None
        self._is_main_process = True
        
        # Statistics
        self._pdb_processed_count = 0
        self._cache_hit_count = 0
        self._pt_file_count = 0
        self._failed_files = set()  # Track files that failed to process
        self._valid_files = None  # Cache of validated files
        
        # Error handling options
        self.skip_invalid_files = kwargs.get('skip_invalid_files', True)
        self.validate_on_init = kwargs.get('validate_on_init', False)  # Pre-validate files during init
        self.max_retry_attempts = kwargs.get('max_retry_attempts', 1)
        
        # ESM-C configuration
        self.esm_use_cpu = kwargs.get('esm_use_cpu', False)
        if self.esm_use_cpu:
            os.environ['ESM_USE_CPU'] = 'true'
            print("Dataset: Configured ESM-C to use CPU only")
        
        # Pre-validate files if requested
        if self.validate_on_init and self.use_pdb_files:
            self._validate_pdb_files()
    
    def __getstate__(self):
        """Custom pickling - exclude locks and worker-specific state."""
        state = self.__dict__.copy()
        # Remove unpickleable entries
        del state['_cache_lock']
        # Reset worker-specific state for new process
        state['_worker_id'] = None
        state['_is_main_process'] = False  # Workers are not main process
        return state
    
    def __setstate__(self, state):
        """Custom unpickling - restore state and create new lock."""
        self.__dict__.update(state)
        # Create new lock in worker process
        self._cache_lock = threading.Lock()
        # Worker will determine its own ID later

    def _determine_data_source(self):
        """Determine whether to use preprocessed .pt files or PDB files."""
        self.use_pdb_files = False
        self.data_source_type = "unknown"
        
        # Check if we should force PDB processing
        if self.force_pdb_processing:
            if os.path.exists(self.pdb_path):
                self.use_pdb_files = True
                self.data_source_type = "pdb_forced"
                print(f"Dataset: Forced PDB processing from {self.pdb_path}")
            else:
                raise ValueError(f"PDB path not found: {self.pdb_path}")
        
        # Check for preprocessed .pt files first (backward compatibility)
        elif self.data_path and os.path.exists(self.data_path):
            pt_files = glob.glob(os.path.join(self.data_path, "*.pt"))
            if pt_files:
                self.use_pdb_files = False
                self.data_source_type = "preprocessed"
                print(f"Dataset: Using preprocessed .pt files from {self.data_path}")
                return
        
        # Fall back to PDB files if enabled
        if self.enable_pdb_processing and os.path.exists(self.pdb_path):
            pdb_files = glob.glob(os.path.join(self.pdb_path, "*.pdb"))
            if pdb_files:
                self.use_pdb_files = True
                self.data_source_type = "pdb_fallback"
                print(f"Dataset: Using PDB files from {self.pdb_path} (no preprocessed files found)")
            else:
                raise ValueError(f"No PDB files found in {self.pdb_path}")
        else:
            if not self.enable_pdb_processing:
                raise ValueError(f"No preprocessed files found in {self.data_path} and PDB processing is disabled")
            else:
                raise ValueError(f"No data files found in {self.data_path} or {self.pdb_path}")

    def _get_file_paths(self):
        """Get all protein file paths based on determined data source."""
        file_paths = []
        
        if self.use_pdb_files:
            # Use PDB files
            if os.path.exists(self.pdb_path):
                pattern = '*.pdb'
                file_paths.extend(glob.glob(os.path.join(self.pdb_path, pattern)))
        else:
            # Use preprocessed .pt files
            if self.data_path and os.path.exists(self.data_path):
                pattern = '*.pt'
                file_paths.extend(glob.glob(os.path.join(self.data_path, pattern)))
            
        # Sort for consistent ordering
        file_paths.sort()
        print(f"Dataset: Found {len(file_paths)} files ({self.data_source_type})")
        return file_paths
    
    def _generate_protein_ids(self):
        """Generate protein IDs from file names"""
        protein_ids = []
        for file_path in self.file_paths:
            filename = Path(file_path).stem  # Get filename without extension
            
            # Handle different filename patterns
            if self.use_pdb_files:
                # For PDB files, use the filename directly
                protein_ids.append(filename)
            else:
                # For .pt files, remove 'pytorch_graph_' prefix if present
                if filename.startswith('pytorch_graph_'):
                    protein_id = filename[len('pytorch_graph_'):]
                else:
                    protein_id = filename
                protein_ids.append(protein_id)
                
        return protein_ids
    
    def _create_data_splits(self):
        """Create train/test splits and select files based on self.train"""
        if not hasattr(self, 'file_paths') or not self.file_paths:
            raise ValueError("File paths are not initialized. Please check data_path.")
        
        # Split the dataset into train and test sets
        train_files, test_files = train_test_split(
            self.file_paths, test_size=0.1, random_state=self.seed)
        
        # if self.train:
        #     self.file_paths = train_files
        # else:
        #     self.file_paths = test_files
        
        # Regenerate protein IDs after split
        self.protein_ids = self._generate_protein_ids()
    
    def _validate_pdb_files(self):
        """Pre-validate PDB files to identify problematic ones."""
        if not self.use_pdb_files:
            return
        
        print(f"Dataset: Pre-validating {len(self.file_paths)} PDB files...")
        valid_files = []
        failed_files = []
        
        for i, file_path in enumerate(self.file_paths):
            if i % 50 == 0:
                print(f"Dataset: Validation progress: {i}/{len(self.file_paths)}")
            
            try:
                if self._is_pdb_file_valid(file_path):
                    valid_files.append(file_path)
                else:
                    failed_files.append(file_path)
                    self._failed_files.add(file_path)
            except Exception as e:
                print(f"Dataset: Validation failed for {file_path}: {e}")
                failed_files.append(file_path)
                self._failed_files.add(file_path)
        
        print(f"Dataset: Validation complete - Valid: {len(valid_files)}, Failed: {len(failed_files)}")
        
        if failed_files:
            print(f"Dataset: Failed files: {[Path(f).name for f in failed_files[:10]]}" + 
                  (f" (and {len(failed_files)-10} more)" if len(failed_files) > 10 else ""))
        
        # Update file paths to only include valid files
        if self.skip_invalid_files:
            self.file_paths = valid_files
            self.protein_ids = self._generate_protein_ids()
            print(f"Dataset: Using {len(self.file_paths)} valid files after filtering")
    
    def _is_pdb_file_valid(self, pdb_file_path):
        """Quick validation check for PDB file without full processing."""
        try:
            from Bio.PDB import PDBParser
            import numpy as np
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("test", pdb_file_path)
            
            # Check if we have any valid chains with coordinates
            valid_residues = 0
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.has_id("CA"):
                            ca_atom = residue["CA"]
                            coords = ca_atom.get_coord()
                            
                            # Check for NaN or infinite coordinates
                            if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                                print(f"Dataset: Invalid coordinates in {pdb_file_path}: {coords}")
                                return False
                            
                            valid_residues += 1
                            
                            # Early exit if we have enough valid residues
                            if valid_residues >= self.min_nodes:
                                return True
            
            if valid_residues < self.min_nodes:
                print(f"Dataset: Too few valid residues in {pdb_file_path}: {valid_residues} < {self.min_nodes}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Dataset: Validation error for {pdb_file_path}: {e}")
            return False

    def _get_worker_info(self):
        """Get current worker information for multiprocessing awareness"""
        try:
            import torch.utils.data as data
            worker_info = data.get_worker_info()
            if worker_info is not None:
                self._worker_id = worker_info.id
                self._is_main_process = False
                # Adjust cache size per worker
                self.cache_size = max(32, self.cache_size // worker_info.num_workers)
            return worker_info
        except:
            return None

    def _get_pdb_processor(self):
        """Get PDB processor with lazy initialization."""
        if self.pdb_processor is None:
            self.pdb_processor = PDBProcessor()
        return self.pdb_processor

    def _load_single_graph(self, file_path):
        """Load a single protein graph with improved thread-safe caching"""
        if not self.enable_cache:
            return self._load_from_disk(file_path)
        
        # Get worker info on first access
        if self._worker_id is None:
            self._get_worker_info()
        
        # Thread-safe cache access
        with self._cache_lock:
            # Check cache first
            if file_path in self._cache:
                # Move to end (most recently used)
                graph_data = self._cache.pop(file_path)
                self._cache[file_path] = graph_data
                return graph_data
        
        # Load from disk if not in cache
        graph_data = self._load_from_disk(file_path)
        
        if graph_data is not None and self.enable_cache:
            self._add_to_cache(file_path, graph_data)
        
        return graph_data
    
    def _load_from_disk(self, file_path):
        """Load graph data from disk or process PDB file on-demand."""
        try:
            if self.use_pdb_files:
                return self._load_from_pdb(file_path)
            else:
                return self._load_from_pt_file(file_path)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _load_from_pt_file(self, file_path):
        """Load graph data from preprocessed .pt file."""
        try:
            graph_data = torch.load(file_path, map_location='cpu')
            
            self._pt_file_count += 1
            return graph_data
            
        except Exception as e:
            print(f"Error loading .pt file {file_path}: {e}")
            return None
    
    def _load_from_pdb(self, pdb_file_path):
        """Load graph data from PDB file with persistent caching."""
        try:
            # Check persistent cache first
            if self.cache_manager.exists(pdb_file_path):
                graph_data = self.cache_manager.load(pdb_file_path)
                if graph_data is not None:
                    self._cache_hit_count += 1
                    return graph_data
            
            # Process PDB file on-the-fly
            print(f"Processing PDB file on-the-fly: {pdb_file_path}")
            processor = self._get_pdb_processor()
            graph_data = processor.process_pdb_file(pdb_file_path)
            
            if graph_data is not None:
                # Save to persistent cache
                self.cache_manager.save(pdb_file_path, graph_data)
                self._pdb_processed_count += 1
                return graph_data
            else:
                print(f"Failed to process PDB file: {pdb_file_path}")
                return None
                
        except Exception as e:
            print(f"Error processing PDB file {pdb_file_path}: {e}")
            return None
        
    def _add_to_cache(self, file_path, graph_data):
        """Add graph to cache with thread-safe LRU eviction"""
        if self.cache_size <= 0:
            # Cache is disabled
            return
            
        with self._cache_lock:
            # Remove oldest items if cache is full
            while len(self._cache) >= self.cache_size:
                if self._cache:  # Check if cache is not empty before popitem
                    self._cache.popitem(last=False)  # Remove oldest (FIFO)
                else:
                    break  # Exit if cache becomes empty
            
            # Add new item (will be most recent)
            self._cache[file_path] = graph_data

    def clear_cache(self):
        """Clear the cache to free memory"""
        with self._cache_lock:
            self._cache.clear()

    def get_cache_info(self):
        """Get cache statistics"""
        with self._cache_lock:
            basic_info = {
                'cache_size': len(self._cache),
                'max_cache_size': self.cache_size,
                'worker_id': self._worker_id,
                'is_main_process': self._is_main_process,
                'cache_enabled': self.enable_cache,
                'data_source_type': self.data_source_type,
                'use_pdb_files': self.use_pdb_files,
                'pdb_processed_count': self._pdb_processed_count,
                'cache_hit_count': self._cache_hit_count,
                'pt_file_count': self._pt_file_count
            }
            
            # Add persistent cache info if using PDB files
            if self.use_pdb_files:
                persistent_cache_info = self.cache_manager.get_cache_stats()
                basic_info.update({
                    'persistent_cache': persistent_cache_info
                })
            
            return basic_info
    
    def get_processing_stats(self):
        """Get detailed processing statistics."""
        return {
            'data_source': self.data_source_type,
            'total_files': len(self.file_paths),
            'failed_files_count': len(self._failed_files),
            'failed_files': [Path(f).name for f in list(self._failed_files)[:10]],  # Show first 10
            'pdb_processed_on_fly': self._pdb_processed_count,
            'persistent_cache_hits': self._cache_hit_count,
            'pt_files_loaded': self._pt_file_count,
            'success_rate': (len(self.file_paths) - len(self._failed_files)) / len(self.file_paths) if len(self.file_paths) > 0 else 0,
            'cache_manager_stats': self.cache_manager.get_cache_stats() if self.use_pdb_files else None
        }
    
    def get_failed_files(self):
        """Get list of files that failed to process."""
        return list(self._failed_files)
    
    def clear_failed_files(self):
        """Clear the list of failed files (useful for retrying)."""
        self._failed_files.clear()
        print("Dataset: Cleared failed files list")

    def warm_cache(self, num_samples=None):
        """Pre-load a subset of the dataset into cache"""
        if not self.enable_cache or not self._is_main_process:
            return
        
        num_samples = num_samples or min(self.cache_size, len(self.file_paths))
        print(f"Warming cache with {num_samples} samples...")
        
        # Load a random subset for better distribution
        indices = random.sample(range(len(self.file_paths)), num_samples)
        
        for i, idx in enumerate(indices):
            file_path = self.file_paths[idx]
            self._load_single_graph(file_path)
            
            if (i + 1) % 50 == 0:
                print(f"Cache warming progress: {i + 1}/{num_samples}")
        
        print(f"Cache warmed. Current cache info: {self.get_cache_info()}")

    def __len__(self):
        # Return dataset size
        return len(self.file_paths)

    def _data_to_dict(self, data_obj):
        """Convert PyTorch Geometric Data object to dictionary format for subgraph sampler."""
        from torch_geometric.data import Data
        
        if isinstance(data_obj, Data):
            return {
                'node_features': data_obj.x,
                'edge_index': data_obj.edge_index, 
                'edge_attr': data_obj.edge_attr,
                'node_pos': data_obj.pos,
                'num_nodes': data_obj.x.shape[0] if data_obj.x is not None else 0
            }
        else:
            # Already dictionary format
            return data_obj
    
    def _dict_to_data(self, dict_obj):
        """Convert dictionary format back to PyTorch Geometric Data object."""
        from torch_geometric.data import Data
        
        if isinstance(dict_obj, dict):
            return Data(
                x=dict_obj['node_features'],
                edge_index=dict_obj['edge_index'],
                edge_attr=dict_obj['edge_attr'],
                pos=dict_obj['node_pos']
            )
        else:
            # Already Data object
            return dict_obj
    
    def _get_num_nodes(self, graph_obj):
        """Get number of nodes from either Data object or dictionary."""
        from torch_geometric.data import Data
        
        if isinstance(graph_obj, Data):
            return graph_obj.x.shape[0] if graph_obj.x is not None else 0
        else:
            return graph_obj.get('num_nodes', graph_obj['node_features'].shape[0])

    def __getitem__(self, idx):
        """Load protein graph on-demand and generate two contrastive views with robust error handling."""
        max_attempts = len(self.file_paths)  # Maximum attempts to find a valid sample
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Get current file (with wraparound if we've tried many times)
                current_idx = (idx + attempt) % len(self.file_paths)
                file_path = self.file_paths[current_idx]
                protein_id = self.protein_ids[current_idx]
                
                # Skip files that we know have failed before
                if self.skip_invalid_files and file_path in self._failed_files:
                    attempt += 1
                    continue
                
                # Check graph file existence, skip to debug
                pt_file_path = '/'.join(file_path.split('/')[:-2]) + '/graph_cache/protein_pdb_files/' + 'pytorch_graph_' + protein_id + '.pt'
                if os.path.exists(pt_file_path) and not self.force_pdb_processing:
                    return {
                        'load_error': False
                    }

                # Load graph data
                protein_graph = self._load_single_graph(file_path)
                
                if protein_graph is None:
                    # Mark this file as failed and try next
                    self._failed_files.add(file_path)
                    print(f"Dataset: Marking {Path(file_path).name} as failed, trying next file...")
                    attempt += 1
                    continue
                
                # Validate graph data
                if not self._validate_graph_data(protein_graph):
                    self._failed_files.add(file_path)
                    print(f"Dataset: Graph validation failed for {Path(file_path).name}, trying next file...")
                    attempt += 1
                    continue
                
                # Check if graph is large enough for subgraph sampling
                num_nodes = self._get_num_nodes(protein_graph)
                if num_nodes < self.min_nodes:
                    self._failed_files.add(file_path)
                    print(f"Dataset: Graph too small for {Path(file_path).name} ({num_nodes} < {self.min_nodes}), trying next file...")
                    attempt += 1
                    continue
                
                # Generate two contrastive views
                view1, view2, sampling_strategy, noise_applied = self._generate_contrastive_views(protein_graph)
                
                # Validate views
                if view1 is None or view2 is None:
                    print(f"Dataset: Failed to generate views for {Path(file_path).name}, trying next file...")
                    attempt += 1
                    continue
                
                # Successfully created a valid sample
                return {
                    'protein_graph': protein_graph,
                    'protein_id': protein_id,
                    'idx': current_idx,
                    'file_path': file_path,
                    'load_error': False,
                    'view1': view1,
                    'view2': view2,
                    'sampling_strategy': sampling_strategy,
                    'noise_applied': noise_applied,
                    'attempts_needed': attempt + 1
                }
                
            except Exception as e:
                print(f"Dataset: Unexpected error processing {self.file_paths[(idx + attempt) % len(self.file_paths)]}: {e}")
                attempt += 1
                continue
        
        # If we get here, we couldn't find any valid sample after many attempts
        print(f"Dataset: Could not find valid sample after {max_attempts} attempts starting from idx {idx}")
        
        # Return a dummy sample to prevent DataLoader from crashing
        return self._create_dummy_sample(idx)
    
    def _validate_graph_data(self, graph_data):
        """Validate that graph data is complete and has no NaN/infinite values."""
        try:
            if graph_data is None:
                return False
            
            # Check for NaN or infinite values
            import torch
            
            # Check node features
            if torch.any(torch.isnan(graph_data.x)) or torch.any(torch.isinf(graph_data.x)):
                print("Dataset: NaN/Inf found in node_features")
                return False
            
            # Check node positions
            if torch.any(torch.isnan(graph_data.pos)) or torch.any(torch.isinf(graph_data.pos)):
                print("Dataset: NaN/Inf found in node_pos")
                return False
            
            # Check edge attributes if present
            if graph_data.edge_attr is not None:
                if torch.any(torch.isnan(graph_data.edge_attr)) or torch.any(torch.isinf(graph_data.edge_attr)):
                    print("Dataset: NaN/Inf found in edge_attr")
                    return False
            
            # Check tensor shapes are consistent
            num_nodes = graph_data.x.shape[0]
            if graph_data.pos.shape[0] != num_nodes:
                print(f"Dataset: Inconsistent shapes - features: {graph_data.x.shape}, pos: {graph_data.pos.shape}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Dataset: Error validating graph data: {e}")
            return False
    
    def _generate_contrastive_views(self, protein_graph):
        """Generate two contrastive views with error handling."""
        try:
            num_nodes = protein_graph.x.shape[0]
            can_sample_subgraph = num_nodes >= self.min_nodes
            
            # Generate two contrastive views with sampling probabilities
            a = getattr(self, 'complete_graph_percent', 0)
            complete_prob = a / 100.0
            seq_prob = (50 - 0.5 * a) / 100.0
            dist_prob = seq_prob
            
            rand_val = random.random()
            
            # Convert Data object to dictionary for subgraph sampler compatibility
            protein_graph_dict = self._data_to_dict(protein_graph)
            
            if not can_sample_subgraph or rand_val < complete_prob:
                # Use complete graph for both views
                view1_dict = self.subgraph_sampler.apply_noise(protein_graph_dict, "identity")
                view2_dict = self.subgraph_sampler.apply_noise(protein_graph_dict, "identity")
                sampling_strategy = 'complete' if can_sample_subgraph else 'complete_forced'
            elif rand_val < complete_prob + seq_prob:
                # Sequential sampling for both views
                view1_dict = self.subgraph_sampler.sample_sequential_subgraph(protein_graph_dict)
                view2_dict = self.subgraph_sampler.sample_sequential_subgraph(protein_graph_dict)
                sampling_strategy = 'sequential'
            else:
                # Distance-based sampling for both views  
                view1_dict = self.subgraph_sampler.sample_distance_subgraph(protein_graph_dict)
                view2_dict = self.subgraph_sampler.sample_distance_subgraph(protein_graph_dict)
                sampling_strategy = 'distance'
            
            # Convert back to Data objects
            view1 = self._dict_to_data(view1_dict)
            view2 = self._dict_to_data(view2_dict)            # Validate views before applying noise
            if not self._validate_graph_data(view1) or not self._validate_graph_data(view2):
                print("Dataset: Invalid views generated during subgraph sampling")
                return None, None, sampling_strategy, 'none'
            
            # Apply noise transformations
            apply_masking = random.random() < 0.5
            
            if apply_masking:
                view1_dict = self._data_to_dict(view1)
                view2_dict = self._data_to_dict(view2)
                view1_dict = self.subgraph_sampler.apply_noise(view1_dict, "random_edge_masking", edge_mask_prob=0.15)
                view2_dict = self.subgraph_sampler.apply_noise(view2_dict, "random_edge_masking", edge_mask_prob=0.15)
                view1 = self._dict_to_data(view1_dict)
                view2 = self._dict_to_data(view2_dict)
                noise_applied = 'masking'
            else:
                view1_dict = self._data_to_dict(view1)
                view2_dict = self._data_to_dict(view2)
                view1_dict = self.subgraph_sampler.apply_noise(view1_dict, "identity")
                view2_dict = self.subgraph_sampler.apply_noise(view2_dict, "identity")
                view1 = self._dict_to_data(view1_dict)
                view2 = self._dict_to_data(view2_dict)
                noise_applied = 'identity'
            
            # Final validation of views
            if not self._validate_graph_data(view1) or not self._validate_graph_data(view2):
                print("Dataset: Invalid views after noise application")
                return None, None, sampling_strategy, noise_applied
            
            return view1, view2, sampling_strategy, noise_applied
            
        except Exception as e:
            print(f"Dataset: Error generating contrastive views: {e}")
            return None, None, 'error', 'none'
    
    def _create_dummy_sample(self, idx):
        """Create a dummy sample when all else fails to prevent DataLoader crashes."""
        return {
            'protein_graph': None,
            'protein_id': f'dummy_{idx}',
            'idx': idx,
            'file_path': 'dummy',
            'load_error': True,
            'view1': None,
            'view2': None,
            'sampling_strategy': 'dummy',
            'noise_applied': 'none',
            'attempts_needed': -1
        }
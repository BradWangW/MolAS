"""
PDB Processing Module for On-the-fly Graph Generation

This module provides functionality to process PDB files into graph representations
with ESM-C embeddings, designed for integration into PyTorch datasets.
"""

import os
import torch
import threading
import numpy as np
import tempfile
from typing import Dict, Optional, Any
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.edges.distance import (
    add_peptide_bonds,
    add_hydrogen_bond_interactions,
    add_disulfide_interactions,
    add_ionic_interactions,
    add_aromatic_interactions,
    add_aromatic_sulphur_interactions,
    add_cation_pi_interactions,
    add_distance_threshold,
    add_sequence_distance_edges,
    add_k_nn_edges,
    node_coords
)
from torch_geometric.data import Data
from functools import partial
import numpy as np
import pandas as pd
from esm.models.esmc import ESMC
import torch
import os

class ESMCSingleton:
    """Thread-safe singleton for ESM-C model management with multiprocessing support."""
    
    _instances = {}  # Process-specific instances
    _locks = {}      # Process-specific locks
    _model = None
    _device = None
    _worker_id = None
    
    def __new__(cls):
        # Get process-specific identifier
        import os
        process_id = os.getpid()
        
        # Create process-specific instance and lock
        if process_id not in cls._instances:
            if process_id not in cls._locks:
                cls._locks[process_id] = threading.Lock()
            
            with cls._locks[process_id]:
                if process_id not in cls._instances:
                    cls._instances[process_id] = super(ESMCSingleton, cls).__new__(cls)
                    # Initialize instance variables
                    instance = cls._instances[process_id]
                    instance._model = None
                    instance._device = None
                    instance._worker_id = None
        
        return cls._instances[process_id]
    
    def __getstate__(self):
        """Custom pickling - exclude locks and model from state."""
        state = self.__dict__.copy()
        # Remove unpickleable entries
        state['_model'] = None  # Don't pickle the model
        state['_device'] = None  # Will be re-determined in worker
        state['_worker_id'] = None  # Will be re-determined in worker
        return state
    
    def __setstate__(self, state):
        """Custom unpickling - restore state and create new lock."""
        self.__dict__.update(state)
        # Model will be loaded lazily when get_model() is called
    
    def get_model(self):
        """Get ESM-C model, loading it if necessary with worker-aware initialization."""
        # Get process-specific lock
        import os
        process_id = os.getpid()
        
        # Ensure process has a lock
        if process_id not in self._locks:
            self._locks[process_id] = threading.Lock()
        
        # Check if we're in a different worker process
        current_worker_id = self._get_worker_id()
        
        if self._model is None or self._worker_id != current_worker_id:
            with self._locks[process_id]:
                if self._model is None or self._worker_id != current_worker_id:
                    self._worker_id = current_worker_id
                    self._load_model()
        return self._model
    
    def _get_worker_id(self):
        """Get current worker ID for multiprocessing awareness."""
        try:
            import torch.utils.data as data
            worker_info = data.get_worker_info()
            return worker_info.id if worker_info is not None else 'main'
        except:
            return 'main'
    
    def _load_model(self):
        """Load ESM-C model with optimal device selection and multiprocessing support."""
        try:
            
            # Set CUDA device for this worker if available
            worker_id = self._worker_id
            
            # Check if CPU-only mode is forced via environment variable
            force_cpu = os.getenv('ESM_USE_CPU', 'false').lower() == 'true'
            
            # Determine best device based on worker and availability
            if torch.cuda.is_available() and not force_cpu:
                # For multiprocessing, we might want to distribute workers across GPUs
                if worker_id != 'main' and isinstance(worker_id, int):
                    # Distribute workers across available GPUs
                    num_gpus = torch.cuda.device_count()
                    if num_gpus > 1:
                        gpu_id = worker_id % num_gpus
                        torch.cuda.set_device(gpu_id)
                        self._device = f"cuda:{gpu_id}"
                    else:
                        self._device = "cuda"
                else:
                    self._device = "cuda"
            else:
                self._device = "cpu"
            
            # Load model
            self._model = ESMC.from_pretrained("esmc_600m").to(self._device)
            self._model.eval()  # Set to evaluation mode
            
        except Exception as e:
            print(f"ESM-C: Failed to load model: {e}")
            
            # Fallback to CPU if GPU initialization fails
            try:
                print("ESM-C: Warning - GPU failed, falling back to CPU...")
                self._device = "cpu"
                self._model = ESMC.from_pretrained("esmc_600m").to(self._device)
                self._model.eval()
            except Exception as e2:
                print(f"ESM-C: Error - CPU fallback also failed: {e2}")
                self._model = None
                self._device = None
                raise
    
    def get_device(self):
        """Get the device the model is on."""
        return self._device
    
    def is_available(self):
        """Check if model is available."""
        return self._model is not None
    
    def reset(self):
        """Reset the singleton (useful for testing or worker cleanup)."""
        import os
        process_id = os.getpid()
        
        # Ensure process has a lock
        if process_id not in self._locks:
            self._locks[process_id] = threading.Lock()
            
        with self._locks[process_id]:
            if self._model is not None:
                try:
                    del self._model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
            self._model = None
            self._device = None
            self._worker_id = None


class PDBProcessor:
    """Process PDB files into graph representations with ESM-C embeddings."""
    
    def __init__(self):
        # Configure protein graph construction
        # Configure Graphein to be robust to non-standard residues and hets.
        # We rely on ESM embeddings for node features, so we omit amino_acid_one_hot
        # to avoid KeyErrors on modified residues (e.g., ALY, MLY) or hets (e.g., 4OG).
        # Also, do not force-include hets to keep the graph protein-only.
        self.config = ProteinGraphConfig(
            granularity="CA",
            node_metadata_functions=[],
            edge_construction_functions=[
                add_peptide_bonds,
                add_aromatic_interactions,
                add_hydrogen_bond_interactions,
                add_disulfide_interactions,
                add_ionic_interactions,
                add_aromatic_sulphur_interactions,
                add_cation_pi_interactions,
                # Additional (GearNet)
                partial(add_distance_threshold, threshold=3.0, long_interaction_threshold=1),
                partial(add_sequence_distance_edges, d=1),
                partial(add_sequence_distance_edges, d=2),
                partial(add_sequence_distance_edges, d=3),
                partial(add_k_nn_edges, k=5),
            ]
        )

        # ESM-C singleton
        self.esm_singleton = ESMCSingleton()
    
    def __getstate__(self):
        """Custom pickling - ensure ESM singleton is properly handled."""
        state = self.__dict__.copy()
        # ESMCSingleton will handle its own pickling
        return state
    
    def __setstate__(self, state):
        """Custom unpickling - restore state."""
        self.__dict__.update(state)
        # ESM singleton will be recreated as needed
    
    def three_to_one(self, resname: str) -> str:
        """Convert three-letter amino acid code to one-letter code."""
        three_to_one_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
            'SEC': 'U', 'PYL': 'O'
        }
        return three_to_one_dict.get(resname, 'X')
    
    def validate_pdb_coordinates(self, pdb_file: str) -> Dict[str, Any]:
        """
        Validate PDB file coordinates before processing with Graphein.
        
        Returns detailed diagnostic information about coordinate validity.
        """
        diagnostics = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'chain_stats': {},
            'coordinate_stats': {
                'total_atoms': 0,
                'valid_atoms': 0,
                'nan_atoms': 0,
                'inf_atoms': 0,
                'missing_coords': 0
            },
            'file_info': {
                'exists': os.path.exists(pdb_file),
                'size_bytes': 0,
                'readable': False
            }
        }
        
        try:
            # Basic file validation
            if not os.path.exists(pdb_file):
                diagnostics['is_valid'] = False
                diagnostics['errors'].append(f"File does not exist: {pdb_file}")
                return diagnostics
            
            file_size = os.path.getsize(pdb_file)
            diagnostics['file_info']['size_bytes'] = file_size
            diagnostics['file_info']['readable'] = os.access(pdb_file, os.R_OK)
            
            if file_size == 0:
                diagnostics['is_valid'] = False
                diagnostics['errors'].append("File is empty")
                return diagnostics
            
            if not diagnostics['file_info']['readable']:
                diagnostics['is_valid'] = False
                diagnostics['errors'].append("File is not readable")
                return diagnostics
            
            # Parse structure using BioPython
            parser = PDBParser(QUIET=True)
            try:
                structure = parser.get_structure("protein", pdb_file)
            except Exception as e:
                diagnostics['is_valid'] = False
                diagnostics['errors'].append(f"Failed to parse PDB structure: {e}")
                return diagnostics
            
            # Analyze each chain
            for model in structure:
                for chain in model:
                    chain_id = chain.id
                    chain_stats = {
                        'residue_count': 0,
                        'atom_count': 0,
                        'valid_atoms': 0,
                        'nan_atoms': 0,
                        'inf_atoms': 0,
                        'missing_coords': 0,
                        'coord_ranges': {'x': [float('inf'), -float('inf')], 
                                       'y': [float('inf'), -float('inf')], 
                                       'z': [float('inf'), -float('inf')]},
                        'has_ca_atoms': 0,
                        'sequence_length': 0
                    }
                    
                    for residue in chain:
                        chain_stats['residue_count'] += 1
                        
                        # Check if residue has CA atom (standard amino acid)
                        if residue.has_id("CA"):
                            chain_stats['has_ca_atoms'] += 1
                            chain_stats['sequence_length'] += 1
                        
                        for atom in residue:
                            chain_stats['atom_count'] += 1
                            diagnostics['coordinate_stats']['total_atoms'] += 1
                            
                            # Get atomic coordinates
                            try:
                                coords = atom.get_coord()
                                
                                if coords is None or len(coords) != 3:
                                    chain_stats['missing_coords'] += 1
                                    diagnostics['coordinate_stats']['missing_coords'] += 1
                                    continue
                                
                                # Check for NaN values
                                if np.any(np.isnan(coords)):
                                    chain_stats['nan_atoms'] += 1
                                    diagnostics['coordinate_stats']['nan_atoms'] += 1
                                    diagnostics['warnings'].append(f"NaN coordinates found in chain {chain_id}, residue {residue.get_id()}, atom {atom.get_id()}")
                                    continue
                                
                                # Check for infinite values
                                if np.any(np.isinf(coords)):
                                    chain_stats['inf_atoms'] += 1
                                    diagnostics['coordinate_stats']['inf_atoms'] += 1
                                    diagnostics['warnings'].append(f"Infinite coordinates found in chain {chain_id}, residue {residue.get_id()}, atom {atom.get_id()}")
                                    continue
                                
                                # Valid coordinates - update statistics
                                chain_stats['valid_atoms'] += 1
                                diagnostics['coordinate_stats']['valid_atoms'] += 1
                                
                                # Update coordinate ranges
                                x, y, z = coords
                                chain_stats['coord_ranges']['x'][0] = min(chain_stats['coord_ranges']['x'][0], x)
                                chain_stats['coord_ranges']['x'][1] = max(chain_stats['coord_ranges']['x'][1], x)
                                chain_stats['coord_ranges']['y'][0] = min(chain_stats['coord_ranges']['y'][0], y)
                                chain_stats['coord_ranges']['y'][1] = max(chain_stats['coord_ranges']['y'][1], y)
                                chain_stats['coord_ranges']['z'][0] = min(chain_stats['coord_ranges']['z'][0], z)
                                chain_stats['coord_ranges']['z'][1] = max(chain_stats['coord_ranges']['z'][1], z)
                                
                            except Exception as e:
                                chain_stats['missing_coords'] += 1
                                diagnostics['coordinate_stats']['missing_coords'] += 1
                                diagnostics['warnings'].append(f"Error getting coordinates for chain {chain_id}, residue {residue.get_id()}, atom {atom.get_id()}: {e}")
                    
                    # Finalize chain statistics
                    if chain_stats['atom_count'] > 0:
                        # Fix infinite ranges if no valid atoms found
                        for coord in ['x', 'y', 'z']:
                            if chain_stats['coord_ranges'][coord][0] == float('inf'):
                                chain_stats['coord_ranges'][coord] = [0.0, 0.0]
                        
                        diagnostics['chain_stats'][chain_id] = chain_stats
                        
                        # Chain-level validation
                        if chain_stats['valid_atoms'] == 0:
                            diagnostics['errors'].append(f"Chain {chain_id}: No valid coordinates found")
                            diagnostics['is_valid'] = False
                        elif chain_stats['nan_atoms'] > 0 or chain_stats['inf_atoms'] > 0:
                            diagnostics['warnings'].append(f"Chain {chain_id}: Contains {chain_stats['nan_atoms']} NaN and {chain_stats['inf_atoms']} infinite coordinates")
                        
                        if chain_stats['has_ca_atoms'] == 0:
                            diagnostics['warnings'].append(f"Chain {chain_id}: No CA atoms found (non-standard residues only)")
            
            # Overall validation
            if diagnostics['coordinate_stats']['total_atoms'] == 0:
                diagnostics['is_valid'] = False
                diagnostics['errors'].append("No atoms found in structure")
            elif diagnostics['coordinate_stats']['valid_atoms'] == 0:
                diagnostics['is_valid'] = False
                diagnostics['errors'].append("No valid coordinates found in any atom")
            elif diagnostics['coordinate_stats']['nan_atoms'] > 0:
                diagnostics['is_valid'] = False
                diagnostics['errors'].append(f"Found {diagnostics['coordinate_stats']['nan_atoms']} atoms with NaN coordinates")
            elif diagnostics['coordinate_stats']['inf_atoms'] > 0:
                diagnostics['is_valid'] = False
                diagnostics['errors'].append(f"Found {diagnostics['coordinate_stats']['inf_atoms']} atoms with infinite coordinates")
            
        except Exception as e:
            diagnostics['is_valid'] = False
            diagnostics['errors'].append(f"Unexpected error during coordinate validation: {e}")
        
        return diagnostics
    
    def validate_node_ordering_consistency(self, graph, chain_sequences: Dict[str, Seq]) -> Dict[str, Any]:
        """
        Validate node ordering consistency between Graphein's internal state and our expectations.
        
        This helps detect the multi-chain node ordering issue described in Graphein PR #220.
        """
        validation_result = {
            'is_consistent': True,
            'warnings': [],
            'chain_analysis': {},
            'total_nodes': len(graph.nodes()),
            'multi_chain': len(chain_sequences) > 1
        }
        
        try:
            # Create our chain-aware mapping
            chain_node_mapping = {}
            node_chain_info = {}
            
            for node in graph.nodes():
                node_data = graph.nodes[node]
                chain_id = node_data.get('chain_id')
                residue_number = node_data.get('residue_number', 0)
                
                if chain_id not in chain_node_mapping:
                    chain_node_mapping[chain_id] = []
                chain_node_mapping[chain_id].append(node)
                node_chain_info[node] = {'chain_id': chain_id, 'residue_number': residue_number}
            
            # Sort nodes by residue number within each chain
            for chain_id in chain_node_mapping:
                chain_node_mapping[chain_id].sort(key=lambda x: graph.nodes[x].get('residue_number', 0))
            
            # Analyze each chain
            for chain_id, nodes in chain_node_mapping.items():
                chain_analysis = {
                    'node_count': len(nodes),
                    'expected_length': len(chain_sequences.get(chain_id, "")),
                    'residue_numbers': [graph.nodes[node].get('residue_number', 0) for node in nodes],
                    'sequential_issues': 0,
                    'coordinate_issues': 0
                }
                
                # Check for non-sequential residue numbers (gaps are normal, but order should be maintained)
                residue_numbers = chain_analysis['residue_numbers']
                for i in range(1, len(residue_numbers)):
                    if residue_numbers[i] < residue_numbers[i-1]:
                        chain_analysis['sequential_issues'] += 1
                
                # Check for coordinate availability
                for node in nodes:
                    coords = node_coords(graph, node)
                    if coords is None or np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                        chain_analysis['coordinate_issues'] += 1
                
                validation_result['chain_analysis'][chain_id] = chain_analysis
                
                # Generate warnings
                if chain_analysis['sequential_issues'] > 0:
                    validation_result['warnings'].append(
                        f"Chain {chain_id}: {chain_analysis['sequential_issues']} non-sequential residue ordering issues"
                    )
                    validation_result['is_consistent'] = False
                
                if chain_analysis['coordinate_issues'] > 0:
                    validation_result['warnings'].append(
                        f"Chain {chain_id}: {chain_analysis['coordinate_issues']} nodes with invalid coordinates"
                    )
                
                if abs(chain_analysis['node_count'] - chain_analysis['expected_length']) > 5:
                    validation_result['warnings'].append(
                        f"Chain {chain_id}: Significant mismatch between graph nodes ({chain_analysis['node_count']}) and sequence length ({chain_analysis['expected_length']})"
                    )
            
            # Multi-chain specific validation
            if validation_result['multi_chain']:
                # Check for potential global residue numbering issues
                all_residue_numbers = []
                for chain_id, nodes in chain_node_mapping.items():
                    all_residue_numbers.extend([graph.nodes[node].get('residue_number', 0) for node in nodes])
                
                # If residue numbers appear to be globally sequential across chains, this might indicate the PR #220 issue
                if len(all_residue_numbers) > 1:
                    sorted_global = sorted(all_residue_numbers)
                    if sorted_global == all_residue_numbers:
                        validation_result['warnings'].append(
                            "Potential global residue numbering detected - may indicate Graphein PR #220 ordering issue"
                        )
                        validation_result['is_consistent'] = False
        
        except Exception as e:
            validation_result['warnings'].append(f"Error during node ordering validation: {e}")
            validation_result['is_consistent'] = False
        
        return validation_result

    def get_pdb_diagnostics(self, pdb_file: str) -> str:
        """
        Get comprehensive diagnostic report for a PDB file.
        
        Returns a formatted string with detailed analysis.
        """
        diagnostics = self.validate_pdb_coordinates(pdb_file)
        
        report = [
            f"\n=== PDB Diagnostic Report: {Path(pdb_file).name} ===",
            f"File Status: {'✓ VALID' if diagnostics['is_valid'] else '✗ INVALID'}",
            f"File Size: {diagnostics['file_info']['size_bytes']} bytes",
            f"Readable: {diagnostics['file_info']['readable']}"
        ]
        
        # Coordinate statistics
        coord_stats = diagnostics['coordinate_stats']
        report.extend([
            "\n--- Coordinate Statistics ---",
            f"Total Atoms: {coord_stats['total_atoms']}",
            f"Valid Atoms: {coord_stats['valid_atoms']} ({coord_stats['valid_atoms']/max(1, coord_stats['total_atoms'])*100:.1f}%)",
            f"NaN Atoms: {coord_stats['nan_atoms']}",
            f"Infinite Atoms: {coord_stats['inf_atoms']}",
            f"Missing Coords: {coord_stats['missing_coords']}"
        ])
        
        # Chain-by-chain analysis
        if diagnostics['chain_stats']:
            report.append("\n--- Chain Analysis ---")
            chain_count = len(diagnostics['chain_stats'])
            report.append(f"Total Chains: {chain_count}")
            
            if chain_count > 1:
                report.append("⚠ Multi-chain protein detected:")
                report.append("  - Susceptible to Graphein PR #220 node ordering issue")
                report.append("  - Centroid calculation may use global residue numbering")
                report.append("  - Our chain-aware processing provides mitigation")
            
            for chain_id, stats in diagnostics['chain_stats'].items():
                coord_ranges = stats['coord_ranges']
                report.extend([
                    f"\nChain {chain_id}:",
                    f"  Residues: {stats['residue_count']}, CA atoms: {stats['has_ca_atoms']}",
                    f"  Total atoms: {stats['atom_count']}, Valid: {stats['valid_atoms']}",
                    f"  Invalid: NaN={stats['nan_atoms']}, Inf={stats['inf_atoms']}, Missing={stats['missing_coords']}",
                    f"  X range: [{coord_ranges['x'][0]:.2f}, {coord_ranges['x'][1]:.2f}]",
                    f"  Y range: [{coord_ranges['y'][0]:.2f}, {coord_ranges['y'][1]:.2f}]",
                    f"  Z range: [{coord_ranges['z'][0]:.2f}, {coord_ranges['z'][1]:.2f}]"
                ])
        
        # Errors and warnings
        if diagnostics['errors']:
            report.append("\n--- ERRORS ---")
            for error in diagnostics['errors']:
                report.append(f"  ✗ {error}")
        
        if diagnostics['warnings']:
            report.append("\n--- WARNINGS ---")
            for warning in diagnostics['warnings'][:10]:  # Limit to first 10 warnings
                report.append(f"  ⚠ {warning}")
            if len(diagnostics['warnings']) > 10:
                report.append(f"  ... and {len(diagnostics['warnings']) - 10} more warnings")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def quick_validate_pdb(self, pdb_file: str) -> bool:
        """
        Quick validation check for PDB files.
        
        Returns True if file passes basic validation, False otherwise.
        Useful for batch processing and filtering.
        """
        try:
            diagnostics = self.validate_pdb_coordinates(pdb_file)
            return diagnostics['is_valid']
        except:
            return False
    
    def create_sanitized_pdb_copy(self, pdb_file: str, output_file: str = None) -> Optional[str]:
        """
        Create a sanitized copy of a PDB file with problematic coordinates removed/fixed.
        
        Args:
            pdb_file: Path to original PDB file
            output_file: Path for sanitized copy (if None, uses temp file)
            
        Returns:
            Path to sanitized file or None if sanitization failed
        """
        try:
            import tempfile
            
            if output_file is None:
                # Create temporary file
                fd, output_file = tempfile.mkstemp(suffix='.pdb', prefix='sanitized_')
                os.close(fd)
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_file)
            
            # Track sanitization statistics
            sanitized_atoms = 0
            removed_atoms = 0
            
            # Process each atom
            for model in structure:
                for chain in model:
                    residues_to_remove = []
                    for residue in chain:
                        atoms_to_remove = []
                        for atom in residue:
                            try:
                                coords = atom.get_coord()
                                if coords is None or len(coords) != 3:
                                    atoms_to_remove.append(atom.id)
                                    removed_atoms += 1
                                    continue
                                
                                # Check for and fix NaN/inf coordinates
                                if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                                    # Replace with average coordinates from nearby atoms
                                    # For now, set to origin (could be improved)
                                    atom.set_coord([0.0, 0.0, 0.0])
                                    sanitized_atoms += 1
                                
                                # Clamp extreme coordinates to reasonable ranges
                                x, y, z = coords
                                x = np.clip(x, -1000, 1000)
                                y = np.clip(y, -1000, 1000) 
                                z = np.clip(z, -1000, 1000)
                                
                                if not np.array_equal(coords, [x, y, z]):
                                    atom.set_coord([x, y, z])
                                    sanitized_atoms += 1
                                    
                            except Exception as e:
                                print(f"Error processing atom {atom.id}: {e}")
                                atoms_to_remove.append(atom.id)
                                removed_atoms += 1
                        
                        # Remove problematic atoms
                        for atom_id in atoms_to_remove:
                            residue.detach_child(atom_id)
                        
                        # If residue has no atoms left, mark for removal
                        if len(residue) == 0:
                            residues_to_remove.append(residue.id)
                    
                    # Remove empty residues
                    for residue_id in residues_to_remove:
                        chain.detach_child(residue_id)
            
            # Write sanitized structure
            from Bio.PDB import PDBIO
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_file)
            
            if sanitized_atoms > 0 or removed_atoms > 0:
                print(f"PDB sanitization completed - Sanitized: {sanitized_atoms}, Removed: {removed_atoms}")
            
            return output_file
            
        except Exception as e:
            print(f"Error sanitizing PDB file: {e}")
            return None

    def get_sequences_from_pdb(self, pdb_file: str) -> Dict[str, Seq]:
        """Extract sequences for all chains from a PDB file."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        chain_sequences = {}
        
        for model in structure:
            for chain in model:
                sequence = []
                for residue in chain:
                    if not residue.has_id("CA"):  # Skip non-standard residues
                        continue
                    sequence.append(residue.get_resname())
                
                chain_id = chain.id
                chain_sequences[chain_id] = Seq("".join([self.three_to_one(res) for res in sequence]))
        
        return chain_sequences
    
    def generate_esm_embeddings(self, chain_sequences: Dict[str, Seq]) -> Dict[str, torch.Tensor]:
        """Generate ESM-C embeddings for protein sequences."""
        if not self.esm_singleton.is_available():
            esm_client = self.esm_singleton.get_model()
        else:
            esm_client = self.esm_singleton.get_model()
        
        chain_embeddings = {}
        
        try:
            from esm.sdk.api import ESMProtein, LogitsConfig
            
            for chain_id, seq in chain_sequences.items():
                protein = ESMProtein(sequence=str(seq))
                protein_tensor = esm_client.encode(protein)
                logits_output = esm_client.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embeddings_trimmed = logits_output.embeddings[:, 1:-1].squeeze(0)  # Remove batch and special tokens
                chain_embeddings[chain_id] = embeddings_trimmed.cpu()  # Move to CPU for storage
                
        except Exception as e:
            print(f"Error generating ESM-C embeddings: {e}")
            raise
        
        return chain_embeddings
    
    def convert_to_pytorch(self, graph, chain_sequences: Dict[str, Seq]) -> Optional[Dict[str, torch.Tensor]]:
        """Convert Graphein protein graph to pytorch tensor objects with robust error handling."""
        try:
            # Generate ESM embeddings for each chain
            chain_embeddings = self.generate_esm_embeddings(chain_sequences)
            
            # Attach sequences to the graph metadata
            graph.graph["sequences"] = chain_sequences
            
            # Multi-chain ordering validation
            if len(chain_sequences) > 1:
                ordering_validation = self.validate_node_ordering_consistency(graph, chain_sequences)
                if not ordering_validation['is_consistent']:
                    print("Warning: Node ordering issues detected - chain-aware mapping will compensate")
                    for warning in ordering_validation['warnings']:
                        print(f"  - {warning}")
            
            # Create sequential mapping for each chain by sorting nodes by residue number
            chain_node_mapping = {}
            for node in graph.nodes():
                node_data = graph.nodes[node]
                chain_id = node_data.get('chain_id')
                if chain_id not in chain_node_mapping:
                    chain_node_mapping[chain_id] = []
                chain_node_mapping[chain_id].append(node)
            
            # Sort nodes by residue number for each chain to create sequential mapping
            # This is crucial for multi-chain proteins to avoid the PR #220 ordering issue
            for chain_id in chain_node_mapping:
                chain_node_mapping[chain_id].sort(key=lambda x: graph.nodes[x].get('residue_number', 0))
            
            # First pass: collect valid nodes (those with coordinates AND embeddings)
            valid_nodes = []
            node_features = []
            node_coordinates = []
            
            for node in graph.nodes():
                node_data = graph.nodes[node]
                chain_id = node_data.get('chain_id')
                
                # Get coordinates
                coords = node_coords(graph, node)
                
                if coords is not None and len(coords) == 3:
                    # Check for NaN or infinite coordinates
                    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                        print(f"Warning: Skipping node {node} with invalid coordinates: {coords}")
                        continue
                    
                    # Use sequential index instead of PDB residue number
                    try:
                        sequential_idx = chain_node_mapping[chain_id].index(node)
                        
                        if chain_id in chain_embeddings and 0 <= sequential_idx < len(chain_embeddings[chain_id]):
                            # Only include nodes that have both valid coordinates and embeddings
                            valid_nodes.append(node)
                            node_coordinates.append(coords)
                            node_features.append(chain_embeddings[chain_id][sequential_idx])
                        else:
                            print(f"Warning: Sequential index {sequential_idx} out of bounds for chain {chain_id}")
                    except ValueError:
                        print(f"Warning: Node {node} not found in chain mapping for chain {chain_id}")
            
            # Create mapping only for valid nodes
            node_mapping = {node: idx for idx, node in enumerate(valid_nodes)}
            
            if len(node_features) == 0:
                print("Warning: No valid nodes found with both coordinates and embeddings")
                return None
            
            # Convert to tensors and validate
            node_features = torch.stack(node_features)
            node_coords_tensor = torch.tensor(node_coordinates, dtype=torch.float32)
            
            # Final validation of tensors
            if torch.any(torch.isnan(node_features)) or torch.any(torch.isinf(node_features)):
                print("Error: NaN/Inf detected in node features")
                return None
            
            if torch.any(torch.isnan(node_coords_tensor)) or torch.any(torch.isinf(node_coords_tensor)):
                print("Error: NaN/Inf detected in node coordinates")
                return None

            # Extract edge indices and attributes - only for edges between valid nodes
            edge_indices = []
            edge_attrs = []
            
            # Separate physio-chemical properties from distance-based metrics
            physio_chemical_mapping = {
                'peptide_bond': [1, 0, 0, 0, 0, 0, 0],
                'hbond': [0, 1, 0, 0, 0, 0, 0],
                'disulfide': [0, 0, 1, 0, 0, 0, 0],
                'ionic': [0, 0, 0, 1, 0, 0, 0],
                'aromatic': [0, 0, 0, 0, 1, 0, 0],
                'aromatic_sulfur': [0, 0, 0, 0, 0, 1, 0],
                'cation_pi': [0, 0, 0, 0, 0, 0, 1],
            }
            
            distance_metric_mapping = {
                'sequence_edge': [1, 0, 0],
                'knn': [0, 1, 0], 
                'distance_threshold': [0, 0, 1],
            }

            for u, v, edge_data in graph.edges(data=True):
                # Only include edges between valid nodes
                if u in node_mapping and v in node_mapping:
                    u_idx = node_mapping[u]
                    v_idx = node_mapping[v]
                    edge_indices.append([u_idx, v_idx])
                    edge_indices.append([v_idx, u_idx])

                    # Handle edge kind - it could be a set or list
                    edge_kind = edge_data.get('kind', set())
                    if isinstance(edge_kind, set):
                        edge_kind = list(edge_kind)
                    
                    # Initialize edge feature vector [physio_chemical_1-7, distance_metric_1-3, distance]
                    physio_chemical_vector = [0, 0, 0, 0, 0, 0, 0]  # Default: no physio-chemical interaction
                    distance_metric_vector = [0, 0, 0]  # Default: no distance-based metric
                    
                    # Categorize edge types - accumulate multiple metrics
                    for kind in edge_kind:
                        if kind in physio_chemical_mapping:
                            # Add the physio-chemical values instead of replacing
                            for i, val in enumerate(physio_chemical_mapping[kind]):
                                physio_chemical_vector[i] += val
                        elif kind in distance_metric_mapping:
                            # Add the distance metric values instead of replacing
                            for i, val in enumerate(distance_metric_mapping[kind]):
                                distance_metric_vector[i] += val
                    
                    dist = edge_data.get('distance', 0.0)
                    
                    # Validate distance value
                    if np.isnan(dist) or np.isinf(dist):
                        dist = 0.0  # Use default distance for invalid values
                    
                    # Create 11D edge feature: [physio_chemical_1-7, distance_metric_1-3, distance]
                    edge_attrs.append(physio_chemical_vector + distance_metric_vector + [dist])
                    edge_attrs.append(physio_chemical_vector + distance_metric_vector + [dist])
            
            if len(edge_indices) == 0:
                print("Warning: No valid edges found")
                # Create minimal connectivity for isolated nodes
                if len(valid_nodes) > 1:
                    # Create a simple chain connectivity
                    for i in range(len(valid_nodes) - 1):
                        edge_indices.append([i, i + 1])
                        edge_indices.append([i + 1, i])
                        # Simple edge features (all zeros except distance)
                        edge_attrs.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1.0])  # Sequential edge
                        edge_attrs.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1.0])
                else:
                    print("Error: Single node graph - cannot create edges")
                    return None
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
            
            # Final validation of edge tensors
            if torch.any(torch.isnan(edge_attr)) or torch.any(torch.isinf(edge_attr)):
                print("Error: NaN/Inf detected in edge attributes")
                return None

            # Validation checks with better error messages
            if node_features.shape[0] != node_coords_tensor.shape[0]:
                print(f"Error: Node features shape {node_features.shape} != coordinates shape {node_coords_tensor.shape}")
                return None
            
            if len(valid_nodes) == 0:
                print("Error: No valid nodes found")
                return None
                
            if edge_index.numel() > 0 and edge_index.max() >= len(valid_nodes):
                print(f"Error: Edge indices max {edge_index.max()} >= num nodes {len(valid_nodes)}")
                return None
                
            if edge_attr.shape[0] != edge_index.shape[1]:
                print(f"Error: Edge attr shape {edge_attr.shape} != edge index shape {edge_index.shape}")
                return None

            # Create Data object with validation
            try:
                result = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pos=node_coords_tensor,
                    num_nodes=len(node_features)
                )
                
                # Verify Data object is valid
                if result.x is None or result.pos is None or result.edge_index is None:
                    print("Error: Data object creation failed - contains None values")
                    return None
                    
                print(f"Success: Created Data object with {result.x.shape[0]} nodes, {result.edge_index.shape[1]} edges")
                
            except Exception as data_error:
                print(f"Error creating Data object: {data_error}")
                return None
            # result = {
            #     'node_features': node_features,
            #     'edge_index': edge_index, 
            #     'edge_attr': edge_attr,
            #     'node_pos': node_coords_tensor,
            #     'num_nodes': len(node_features)
            # }
            
            # Final comprehensive validation
            if not self._validate_final_graph(result):
                print("Error: Final graph validation failed")
                return None
            
            return result
            
        except Exception as e:
            print(f"Error converting graph to PyTorch format: {e}")
            return None
    
    def _validate_final_graph(self, graph_data) -> bool:
        """Perform final validation of PyTorch Geometric Data object."""
        try:
            import torch
            import numpy as np
            from torch_geometric.data import Data
            
            # Check if it's a Data object
            if not isinstance(graph_data, Data):
                print(f"Error: Expected Data object, got {type(graph_data)}")
                return False
            
            # Check for required attributes
            if graph_data.x is None or graph_data.pos is None or graph_data.edge_index is None:
                print("Error: Data object missing required attributes (x, pos, edge_index)")
                return False
            
            # Check for NaN/Inf in all tensors
            if torch.any(torch.isnan(graph_data.x)) or torch.any(torch.isinf(graph_data.x)):
                print("Error: NaN/Inf found in node features")
                return False
                
            if torch.any(torch.isnan(graph_data.pos)) or torch.any(torch.isinf(graph_data.pos)):
                print("Error: NaN/Inf found in node positions")
                return False
                
            if graph_data.edge_attr is not None:
                if torch.any(torch.isnan(graph_data.edge_attr)) or torch.any(torch.isinf(graph_data.edge_attr)):
                    print("Error: NaN/Inf found in edge attributes")
                    return False
            
            # Check tensor shapes consistency
            num_nodes = graph_data.x.shape[0]
            if graph_data.pos.shape[0] != num_nodes:
                print(f"Error: Node features {graph_data.x.shape} != positions {graph_data.pos.shape}")
                return False

            if graph_data.edge_index.numel() > 0:
                num_edges = graph_data.edge_index.shape[1]
                if graph_data.edge_attr is not None and graph_data.edge_attr.shape[0] != num_edges:
                    print(f"Error: Edge attributes {graph_data.edge_attr.shape} != edge index {graph_data.edge_index.shape}")
                    return False
                
                # Check edge indices are within valid range
                if graph_data.edge_index.max() >= num_nodes or graph_data.edge_index.min() < 0:
                    print(f"Error: Edge indices out of range [0, {num_nodes}): max={graph_data.edge_index.max()}, min={graph_data.edge_index.min()}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error in final validation: {e}")
            return False
    
    def process_pdb_file(self, pdb_file_path: str):
        """
        Process a single PDB file into a PyTorch Geometric Data object with comprehensive diagnostics.
        
        Args:
            pdb_file_path: Path to the PDB file
            
        Returns:
            PyTorch Geometric Data object or None if processing failed
        """
        try:
            if not os.path.exists(pdb_file_path):
                print(f"Error: PDB file not found: {pdb_file_path}")
                return None
            
            # Step 1: Pre-validate PDB coordinates
            coord_diagnostics = self.validate_pdb_coordinates(pdb_file_path)
            
            if not coord_diagnostics['is_valid']:
                print(f"Error: PDB validation failed for {Path(pdb_file_path).name}")
                print(self.get_pdb_diagnostics(pdb_file_path))
                return None
            
            if coord_diagnostics['warnings']:
                print(f"Warning: {len(coord_diagnostics['warnings'])} validation warnings for {Path(pdb_file_path).name}")
            
            # Step 2: Extract sequences for all chains
            print('Starting sequence extraction...')
            chain_sequences = self.get_sequences_from_pdb(pdb_file_path)
            
            if not chain_sequences:
                print(f"Error: No valid sequences found in {pdb_file_path}")
                return None
            
            # Step 3: Construct graph with detailed error handling
            try:
                # Try with more permissive settings first
                print('Starting graph construction with Graphein...')
                graph = construct_graph(config=self.config, path=pdb_file_path)
                
                # Post-construction validation of node coordinates
                invalid_nodes = []
                for node in graph.nodes():
                    coords = node_coords(graph, node)
                    if coords is not None and len(coords) == 3:
                        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                            invalid_nodes.append((node, coords))
                
                if invalid_nodes:
                    print(f"Warning: Found {len(invalid_nodes)} nodes with invalid coordinates in constructed graph")
                
                # Validate node ordering consistency (multi-chain issue detection)
                ordering_validation = self.validate_node_ordering_consistency(graph, chain_sequences)
                
                if not ordering_validation['is_consistent']:
                    print(f"Warning: Node ordering consistency issues detected")
                    for warning in ordering_validation['warnings']:
                        print(f"  - {warning}")
                    
                    if ordering_validation['multi_chain']:
                        print("  Multi-chain protein may be affected by Graphein PR #220 issue")
            
            except Exception as graphein_error:
                print(f"Error: Graphein graph construction failed: {graphein_error}")
                
                # Try fallback with simplified configuration
                if len(chain_sequences) > 1:
                    fallback_edges = [
                        add_peptide_bonds,  # Essential for protein backbone
                        partial(add_distance_threshold, threshold=6.0, long_interaction_threshold=1),  # Conservative distance
                    ]
                else:
                    fallback_edges = [
                        add_peptide_bonds,
                        add_aromatic_interactions,
                        add_hydrogen_bond_interactions,
                        add_disulfide_interactions,
                        add_ionic_interactions,
                        add_aromatic_sulphur_interactions,
                        add_cation_pi_interactions,
                        partial(add_distance_threshold, threshold=6.0, long_interaction_threshold=1),
                    ]
                
                try:
                    # Use no amino acid one-hot in fallback to avoid crashes on modified residues
                    simplified_config = ProteinGraphConfig(
                        granularity="CA",
                        node_metadata_functions=[],
                        edge_construction_functions=fallback_edges
                    )
                    
                    graph = construct_graph(config=simplified_config, path=pdb_file_path)
                    print(f"Warning: Using simplified fallback configuration for {Path(pdb_file_path).name}")
                    
                except Exception as fallback_error:
                    print(f"Error: Fallback also failed: {fallback_error}")
                    print("This PDB file is fundamentally incompatible with Graphein processing")
                    return None
            
            # Step 4: Convert to PyTorch format
            pytorch_graph = self.convert_to_pytorch(graph, chain_sequences)
            
            if pytorch_graph is None:
                print(f"Error: Failed to convert {pdb_file_path} to PyTorch format")
                return None
            
            return pytorch_graph
            
        except Exception as e:
            print(f"Error: Unexpected error processing {pdb_file_path}: {e}")
            return None

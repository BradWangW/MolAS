import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import math
import numpy as np

#---------------------------Change dataset-----------------------------
# These will be set dynamically via function parameters
path = '/data1/home/jw1017/GLAS_Dock/Data/ready_data'

ligand_graph_path = f'{path}/ligand_graphs'
protein_graph_path = f'{path}/protein_graphs_esmc_600m'
#----------------------------------------------------------------------

def get_dataset_paths(benchmark='posex_self_docking'):
    """Generate dataset paths based on benchmark and relaxation settings"""
    pb_ratios_path = f'{path}/{benchmark}_pb_ratio.csv'
    rmsd_path = f'{path}/{benchmark}_rmsd.csv'
    
    return pb_ratios_path, rmsd_path

def pb_score(x):
    
    # binary score
    return 1 if x == 1 else 0
    
# RMSD scores
def rmsd_score(x, lambda_rmsd=3):
    M = 2
    if x > 5 or x < 0:
        return 0
    return (1 + math.exp(-lambda_rmsd * M)) / (1 + math.exp(lambda_rmsd * (x - M)))

# Dataset classes
# ligand graph
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.ligand_graph_path = ligand_graph_path
                    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        ligand_name = self.dataframe.iloc[idx]['ligand']
        protein_name = self.dataframe.iloc[idx]['protein']
        names = f'{protein_name}_{ligand_name}'
        
        # Direct loading without cache for better performance
        graph_path = f'{self.ligand_graph_path}/pyg_graph_{names}.pt'
        nodes, edge_index, edge_attr, pos = torch.load(graph_path, weights_only=False, map_location='cpu')
        
        # Create data with explicit memory optimization
        graph_data = Data(
            x=nodes[1].to(torch.float32).contiguous(),  # Ensure contiguous memory layout and dtype
            edge_index=edge_index[1].contiguous(), 
            edge_attr=edge_attr[1].to(torch.float32).contiguous(), 
            pos=pos[1].to(torch.float32).contiguous()
        )
        graph_data.names = names
        return graph_data

# Dataset classes
# ligand graph
class SMILESDatasetChemberta(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        lig_chemberta = torch.load(f"{path}/ligand_chemberta.pt")
        self.lig_emb_map = dict(zip(lig_chemberta["ligand_ids"],
                                    lig_chemberta["embeddings"]))
                    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        ligand_name = self.dataframe.iloc[idx]['ligand']
        protein_name = self.dataframe.iloc[idx]['protein']
        names = f'{protein_name}_{ligand_name}'
        
        # Create data with explicit memory optimization
        graph_data = Data(
            x=self.lig_emb_map[names].to(torch.float32).contiguous()
        )
        graph_data.names = names
        return graph_data
    
# protein graph
class GraphDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.protein_graph_path = protein_graph_path
        # Cache for loaded graphs to prevent memory leaks
        self._graph_cache = {}
                    
    def __len__(self):
        return len(self.dataframe)
        
    def validate_graph(self, x, pos, edge_index, edge_attr=None):
        N = pos.size(0)
        assert x.size(0) == N
        if edge_index.numel():
            m, n = int(edge_index.max()), int(edge_index.min())
            assert 0 <= n and m < N
        if edge_attr is not None:
            assert edge_attr.size(0) == edge_index.size(1)
    
    def __getitem__(self, idx):
        protein_name = self.dataframe.iloc[idx]['protein']
        ligand_name = self.dataframe.iloc[idx]['ligand']
        names = f'{protein_name}_{ligand_name}'
        
        # Use cache to avoid reloading the same graph
        if names not in self._graph_cache:
            graph_path = f'{self.protein_graph_path}/pyg_graph_{names}_esmc_600m.pt'
            # graph_path = f'{self.protein_graph_path}/pytorch_graph_{names}_protein.pt'
            graph = torch.load(graph_path, weights_only=False)
            x, pos, edge_index, edge_attr = graph.x, graph.pos, graph.edge_index, graph.edge_attr
            x = x.float()
            self._graph_cache[names] = (x, pos, edge_index, edge_attr)
        else:
            x, pos, edge_index, edge_attr = self._graph_cache[names]
            x = x.float()

        self.validate_graph(x, pos, edge_index, edge_attr)
        
        # print(f'Shapes - x: {x.shape}, pos: {pos.shape}, edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape}')
        # print(f'Type - x: {x.dtype}, pos: {pos.dtype}, edge_index: {edge_index.dtype}, edge_attr: {edge_attr.dtype}')
        
        # Create data with explicit memory optimization  
        graph_data = Data(
            x=x.to(torch.float32).contiguous(),  # Ensure contiguous memory layout and dtype
            pos=pos.to(torch.float32).contiguous(),
            edge_index=edge_index.contiguous(), 
            edge_attr=edge_attr.to(torch.float32).contiguous()  # Slice and make contiguous
        )
        graph_data.names = names
        return graph_data
    
class LabelDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.final_df)

    def __getitem__(self, idx):
        label = self.dataframe.loc[idx, "label"].astype(float)

        return torch.tensor(label, dtype=torch.float32)

# Create Combined Dataset for use in stacked model
class CombinedDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles_dataset = SMILESDatasetChemberta(dataframe)
        # self.smiles_dataset = SMILESDataset(dataframe)
        self.graph_dataset = GraphDataset(dataframe)
        self.label_dataset = LabelDataset(dataframe)

    def __len__(self):
        return len(self.smiles_dataset)

    def __getitem__(self, idx):
        # Get data from both datasets
        smiles_data = self.smiles_dataset[idx]
        graph_data = self.graph_dataset[idx]
        label_data = self.label_dataset[idx]
        
        # Since both dataframes are the same, ligand names should match
        assert smiles_data.names == graph_data.names, "names do not match."

        return smiles_data, graph_data, label_data

def prepare_data_composite(incl_columns:list = [], benchmark='posex_self_docking'):
    # Load the 'pb_ratios' and 'rmsd' datasets
    pb_ratios_path, rmsd_path = get_dataset_paths(benchmark)
    pb_ratios = pd.read_csv(pb_ratios_path)
    rmsd = pd.read_csv(rmsd_path)

    # Merge into a single DataFrame
    protein = [i.split('_')[0] for i in rmsd['PDB_CCD_ID']]
    ligand = ['_'.join(i.split('_')[1:]) for i in rmsd['PDB_CCD_ID']]
    
    rmsd.insert(0, 'protein', protein)
    rmsd.insert(1, 'ligand', ligand)
    rmsd = rmsd.drop(columns=['PDB_CCD_ID'])
    rmsd.columns=['protein', 'ligand'] + ['_'.join(col_name.split('_')[:-1]) for col_name in rmsd.columns[2:]]
    
    pb_ratios.insert(0, 'protein', protein)
    pb_ratios.insert(1, 'ligand', ligand)
    pb_ratios = pb_ratios.drop(columns=['PDB_CCD_ID'])
    pb_ratios.columns=['protein', 'ligand'] + ['_'.join(col_name.split('_')[:-2]) for col_name in pb_ratios.columns[2:]]
    
    pb_ratios = pb_ratios[rmsd.columns] # important: align columns
    
    # Score computation
    pb_scores = pb_ratios.iloc[:, 2:].map(pb_score)
    rmsd_scores = rmsd.iloc[:, 2:].map(rmsd_score)
    
    final_df = pb_scores * rmsd_scores
    final_df.insert(0, 'protein', protein)
    final_df.insert(1, 'ligand', ligand)

    # Filter to keep only the specified columns(algorithms)
    if len(incl_columns) > 0:
        final_df = final_df[final_df.columns.intersection(incl_columns+['protein', 'ligand'])]

    # final_df.iloc[:, 2:] = np.random.rand(final_df.shape[0], final_df.shape[1]-2)  # Dummy features for testing
        
    label = [final_df.iloc[i,2:].values for i in range(len(final_df))]
    final_df['label'] = label
    num_classes = len(final_df.columns[2:-1])
    
    # Health check of the final dataframe
    assert not final_df.isnull().values.any(), "DataFrame contains NaN values."
    assert not final_df.isin([float('inf'), float('-inf')]).values.any(), "DataFrame contains infinite values."
    assert all(isinstance(t, (list, np.ndarray)) for t in final_df['label']), "label column contains non-list/array values."
    assert all(len(t) == num_classes for t in final_df['label']), "Inconsistent label lengths."

    return final_df, num_classes

# Directly return the RMSD scores; Currently used for evaluation only!
def prepare_data_rmsd(incl_columns:list = [], benchmark='posex_self_docking'):
    # Load the 'pb_ratios' and 'rmsd' datasets
    pb_ratios_path, rmsd_path = get_dataset_paths(benchmark)
    pb_ratios = pd.read_csv(pb_ratios_path)
    rmsd = pd.read_csv(rmsd_path)

    # Merge into a single DataFrame
    protein = [i.split('_')[0] for i in rmsd['PDB_CCD_ID']]
    ligand = ['_'.join(i.split('_')[1:]) for i in rmsd['PDB_CCD_ID']]
    
    rmsd.insert(0, 'protein', protein)
    rmsd.insert(1, 'ligand', ligand)
    rmsd = rmsd.drop(columns=['PDB_CCD_ID'])
    rmsd.columns=['protein', 'ligand'] + ['_'.join(col_name.split('_')[:-1]) for col_name in rmsd.columns[2:]]
    
    pb_ratios.insert(0, 'protein', protein)
    pb_ratios.insert(1, 'ligand', ligand)
    pb_ratios = pb_ratios.drop(columns=['PDB_CCD_ID'])
    pb_ratios.columns=['protein', 'ligand'] + ['_'.join(col_name.split('_')[:-2]) for col_name in pb_ratios.columns[2:]]
    
    pb_ratios = pb_ratios[rmsd.columns] # important: align columns

    final_df = rmsd

    # Filter to keep only the specified columns(algorithms)
    if len(incl_columns) > 0:
        final_df = final_df[final_df.columns.intersection(incl_columns+['protein', 'ligand'])]

    label = [final_df.iloc[i,2:].values for i in range(len(final_df))]
    final_df['label'] = label
    num_classes = len(final_df.columns[2:-1])

    return final_df, num_classes

# Directly return the PoseBuster pass (or not); Currently used for evaluation only!
def prepare_data_pose(incl_columns:list = [], benchmark='posex_self_docking'):
    # Load the 'pb_ratios' and 'rmsd' datasets
    pb_ratios_path, rmsd_path = get_dataset_paths(benchmark)
    pb_ratios = pd.read_csv(pb_ratios_path)
    rmsd = pd.read_csv(rmsd_path)

    # Merge into a single DataFrame
    protein = [i.split('_')[0] for i in rmsd['PDB_CCD_ID']]
    ligand = ['_'.join(i.split('_')[1:]) for i in rmsd['PDB_CCD_ID']]
    
    rmsd.insert(0, 'protein', protein)
    rmsd.insert(1, 'ligand', ligand)
    rmsd = rmsd.drop(columns=['PDB_CCD_ID'])
    rmsd.columns=['protein', 'ligand'] + ['_'.join(col_name.split('_')[:-1]) for col_name in rmsd.columns[2:]]
    
    pb_ratios.insert(0, 'protein', protein)
    pb_ratios.insert(1, 'ligand', ligand)
    pb_ratios = pb_ratios.drop(columns=['PDB_CCD_ID'])
    pb_ratios.columns=['protein', 'ligand'] + ['_'.join(col_name.split('_')[:-2]) for col_name in pb_ratios.columns[2:]]
    
    pb_ratios = pb_ratios[rmsd.columns] # important: align columns

    final_df = pb_ratios.iloc[:, 2:].map(pb_score).astype(int) # True if PoseBuster passes, False otherwise
    final_df.insert(0, 'protein', protein)
    final_df.insert(1, 'ligand', ligand)

    # Filter to keep only the specified columns(algorithms)
    if len(incl_columns) > 0:
        final_df = final_df[final_df.columns.intersection(incl_columns+['protein', 'ligand'])]

    label = [final_df.iloc[i,2:].values for i in range(len(final_df))]
    final_df['label'] = label
    num_classes = len(final_df.columns[2:-1])

    return final_df, num_classes
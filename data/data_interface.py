import random
import pandas as pd

import pytorch_lightning as pl
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, DataListLoader
from .dataset import CombinedDataset, prepare_data_composite, protein_graph_path
from sklearn.model_selection import train_test_split, KFold
from torch_geometric.utils import degree
import torch
from tqdm import tqdm

class DInterface(pl.LightningDataModule):
    def __init__(self, num_workers=8, dataset='', **kwargs):
        super().__init__()
        pl.seed_everything(kwargs.get('seed'), workers=True)
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.batch_size = kwargs.get('batch_size', 8)
        self.incl_columns = kwargs.get('incl_columns', [])
        self.test_size = kwargs.get('test_size', 0.5)
        self.seed = kwargs.get('seed')
        self.k_folds = kwargs.get('k_folds', None)  # None means no K-Fold, just train/val/test split
        self.current_fold = kwargs.get('fold_num', 0)  # Default to first fold
        self.benchmark = kwargs.get('benchmark', 'posex_self_docking')

        # Prepare dataset differently based on AS criteria
        if isinstance(self.benchmark, list):
            
            for b in self.benchmark:
                ds, num_classes = prepare_data_composite(self.incl_columns, b)
                
                if hasattr(self, 'dataset'):
                    self.dataset = pd.concat([self.dataset, ds], ignore_index=True)
                else:
                    self.dataset = ds
                    self.num_classes = num_classes
                
        elif isinstance(self.benchmark, str):
            self.dataset, self.num_classes = prepare_data_composite(self.incl_columns, self.benchmark)

        else:
            raise ValueError("benchmark must be a string or a list of strings")
            
        self.load_data_module()

    def setup(self, stage=None):
        if self.k_folds is not None:
            print("K-Fold enabled")            
            self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            self.folds = list(self.kf.split(self.dataset))
            train_idx, test_idx = self.folds[self.current_fold]
            
            self.trainset, self.valset = train_test_split(self.dataset.iloc[train_idx], test_size=0.1, random_state=self.seed)
            self.testset = self.dataset.iloc[test_idx]

        else:
            print("K-Fold disabled")
            if self.test_size == 1 or self.test_size == 0:
                self.trainset, self.valset = train_test_split(self.dataset, test_size=0.1, random_state=self.seed)
                self.testset = self.dataset
            else:
                train_val, self.testset = train_test_split(self.dataset, test_size=self.test_size, random_state=self.seed)
                self.trainset, self.valset = train_test_split(train_val, test_size=self.test_size, random_state=self.seed)

    def train_dataloader(self):
        # Combining the protein and ligand graphs into the dataset
        trainset_combined = CombinedDataset(self.trainset)
        return DataLoader(trainset_combined, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=False, drop_last=False)

    def val_dataloader(self):
        valset_combined = CombinedDataset(self.valset)
        return DataLoader(valset_combined, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=False, drop_last=False)

    def test_dataloader(self):
        testset_combined = CombinedDataset(self.testset)
        return DataLoader(testset_combined, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=False)
    
    def get_train_deg(self):

        deg_hist = torch.zeros(256, dtype=torch.long)
        return deg_hist 
        graph_ds = self.trainset  # or self.graph_dataset if that’s what you saved

        for idx in range(len(graph_ds)):
            # Load just the graph part directly

            protein_name = graph_ds.iloc[idx]['protein']
            ligand_name = graph_ds.iloc[idx]['ligand']
            graph_path = f'{protein_graph_path}/pyg_graph_{protein_name}_{ligand_name}_protein.pt'
            graph = torch.load(graph_path, weights_only=False)
            d = degree(graph.edge_index[1], num_nodes=graph.x.size(0), dtype=torch.long)
            deg_hist.index_add_(0, d, torch.ones_like(d))

        return deg_hist

    # These are from the template, but not used in our implementation.
    def load_data_module(self):
        pass

    def instancialize(self, **other_args):
        pass
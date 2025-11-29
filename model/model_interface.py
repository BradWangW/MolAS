import torch
import numpy as np
import importlib
from torch.nn import functional as F

from utils import ndcg_score

from model.ndcg_loss import LambdaNDCGLoss2
from model.additive_loss import PairwiseLogisticLoss
from model.inspect_tools import linear_CKA, mean_pairwise_cosine

import pytorch_lightning as pl

#TODO: Implement seperately for binary and point-based models
class MInterface(pl.LightningModule):
    def __init__(self, deg, loss, lr, **kargs):
        super().__init__()
        pl.seed_everything(kargs.get('seed'), workers=True)
        self.save_hyperparameters()
        self.deg = deg
        self.load_model()
        self.configure_loss()
        self.ndcg_loss=LambdaNDCGLoss2(sigma=kargs.get('sigma_ndcg', 1))
        self.logistic_loss = PairwiseLogisticLoss(sigma=kargs.get('sigma_logistic', 1))

    def forward(self, ligand_data, protein_data, return_features=False):
        if not return_features:
            return self.model(ligand_data, protein_data)
        return self.model(ligand_data, protein_data, return_features=return_features)

    def training_step(self, batch, batch_idx):
        ligand_data, protein_data, label = batch

        # Get model prediction and bce loss
        prediction = self(ligand_data, protein_data).view(-1, self.hparams.num_classes)
        label = label.view(-1, self.hparams.num_classes)
        loss = self.loss_function(prediction, label)
        
        # Create query_lengths tensor
        query_lengths = torch.full(
            size=(prediction.shape[0],),
            fill_value=self.hparams.num_classes,
            dtype=torch.long,
            device=prediction.device
        )

        # Add NDCG loss if enabled        
        if getattr(self.hparams, 'use_ndcg_loss', False):
            ndcg_loss = compute_rank_loss(
                loss_function=self.ndcg_loss,
                predictions=prediction,
                labels=label,
                query_lengths=query_lengths
            )
            loss += ndcg_loss * getattr(self.hparams, 'ndcg_loss_weight', 1.0)
        
        # Add logistic loss if enabled
        if getattr(self.hparams, 'use_logistic_loss', False):
            logistic_loss = compute_rank_loss(
                loss_function=self.logistic_loss,
                predictions=prediction,
                labels=label,
                query_lengths=query_lengths
            )
            loss += logistic_loss * getattr(self.hparams, 'logistic_loss_weight', 0.01)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Log the total and individual losses
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=False)

        # self.log('gn_egnn', grad_norm(self.model.EGNN_encoder), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=False)
        # self.log('gn_gat', grad_norm(self.model.GAT_encoder), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=False)
        # self.log('gn_gin', grad_norm(self.model.GIN_encoder), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=False)
        # # self.log('gn_egnn_ligand', grad_norm(self.model.ligand_encoder), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=False)
        
        try:
            self.log('gn_res', grad_norm(self.model.Res_decoder), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=False)
        except:
            pass

        # self.log('gn_minimal_processor', grad_norm(self.model.processor), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=False)

        # x_egnn, x_gat = self.model(ligand_data, protein_data, return_gl_outs=True)
        # Compute and log CKA between different GNN embeddings
        # cka_egnn_gat = linear_CKA(x_egnn.detach(), x_gat.detach())
        # cka_egnn_gin = linear_CKA(x_egnn.detach(), x_gin.detach())
        # cka_gat_gin = linear_CKA(x_gat.detach(), x_gin.detach())
        # self.log('cka_egnn_gat', cka_egnn_gat, on_step=True, on_epoch=True, prog_bar=False, batch_size=self.hparams.batch_size, sync_dist=False)
        # self.log('cka_egnn_gin', cka_egnn_gin, on_step=True, on_epoch=True, prog_bar=False, batch_size=self.hparams.batch_size, sync_dist=False)
        # self.log('cka_gat_gin', cka_gat_gin, on_step=True, on_epoch=True, prog_bar=False, batch_size=self.hparams.batch_size, sync_dist=False)

        return loss

    def validation_step(self, batch, batch_idx):
        ligand_data, protein_data, label = batch

        # Get model prediction and bce loss
        prediction = self(ligand_data, protein_data).view(-1, self.hparams.num_classes)
        label = label.view(-1, self.hparams.num_classes)
        loss = self.loss_function(prediction, label)
        
        # Create query_lengths tensor
        query_lengths = torch.full(
            size=(prediction.shape[0],),
            fill_value=self.hparams.num_classes,
            dtype=torch.long,
            device=prediction.device
        )

        # Add NDCG loss if enabled        
        if getattr(self.hparams, 'use_ndcg_loss', False):
            ndcg_loss = compute_rank_loss(
                loss_function=self.ndcg_loss,
                predictions=prediction,
                labels=label,
                query_lengths=query_lengths
            )
            loss += ndcg_loss * getattr(self.hparams, 'ndcg_loss_weight', 1.0)
        
        # Add logistic loss if enabled
        if getattr(self.hparams, 'use_logistic_loss', False):
            logistic_loss = compute_rank_loss(
                loss_function=self.logistic_loss,
                predictions=prediction,
                labels=label,
                query_lengths=query_lengths
            )
            loss += logistic_loss * getattr(self.hparams, 'logistic_loss_weight', 0.01)

        batch_ndcg = []
        for i in range(prediction.shape[0]):
            ndcg = ndcg_score(label[i], prediction[i], self.hparams.ndcg_k)
            batch_ndcg.append(ndcg)
        as_acc = torch.tensor(batch_ndcg, device=loss.device).float().mean()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log('val_as_acc', as_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        if getattr(self.hparams, 'use_ndcg_loss', False):
            self.log('val_ndcg_loss', ndcg_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=self.hparams.batch_size)
        if getattr(self.hparams, 'use_logistic_loss', False):
            self.log('val_logistic_loss', logistic_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=self.hparams.batch_size)

        opt = self.optimizers() if hasattr(self, "optimizers") else None
        lr_value = opt.param_groups[0]['lr'] if opt is not None else None
        if lr_value is not None:
            self.log('lr', lr_value, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=True)

        return {
            'val_loss': loss,
            'val_as_acc': as_acc,
            'lr': torch.tensor(lr_value, device=loss.device) if lr_value is not None else torch.tensor(float('nan'), device=loss.device)
        }

    def test_step(self, batch, batch_idx):
        ligand_data, protein_data, label = batch

        # Get model prediction and bce loss
        prediction = self(ligand_data, protein_data).view(-1, self.hparams.num_classes)
        label = label.view(-1, self.hparams.num_classes)
        loss = self.loss_function(prediction, label)
        
        # Create query_lengths tensor
        query_lengths = torch.full(
            size=(prediction.shape[0],),
            fill_value=self.hparams.num_classes,
            dtype=torch.long,
            device=prediction.device
        )

        # Add NDCG loss if enabled        
        if getattr(self.hparams, 'use_ndcg_loss', False):
            ndcg_loss = compute_rank_loss(
                loss_function=self.ndcg_loss,
                predictions=prediction,
                labels=label,
                query_lengths=query_lengths
            )
            loss += ndcg_loss * getattr(self.hparams, 'ndcg_loss_weight', 1.0)
        
        # Add logistic loss if enabled
        if getattr(self.hparams, 'use_logistic_loss', False):
            logistic_loss = compute_rank_loss(
                loss_function=self.logistic_loss,
                predictions=prediction,
                labels=label,
                query_lengths=query_lengths
            )
            loss += logistic_loss * getattr(self.hparams, 'logistic_loss_weight', 0.01)

        batch_ndcg = []
        for i in range(prediction.shape[0]):
            ndcg = ndcg_score(label[i], prediction[i], self.hparams.ndcg_k)
            batch_ndcg.append(ndcg)
        as_acc = torch.tensor(batch_ndcg, device=loss.device).float().mean()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log('test_as_acc', as_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        if getattr(self.hparams, 'use_ndcg_loss', False):
            self.log('test_ndcg_loss', ndcg_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=self.hparams.batch_size)
        if getattr(self.hparams, 'use_logistic_loss', False):
            self.log('test_logistic_loss', logistic_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=self.hparams.batch_size)

        names = protein_data.names if hasattr(protein_data, 'names') else None
        names_array = np.array(names, dtype=object) if names is not None else None

        output_dict = {
            'test_loss': loss,
            'test_as_acc': as_acc,
            'out': prediction.detach().cpu().numpy(),
            'labels': label.detach().cpu().numpy(),
            'names': names_array,
        }

        if hasattr(self, "test_step_outputs"):
            cpu_output_dict = {
                'test_loss': float(loss.detach().cpu()),
                'test_as_acc': float(as_acc.detach().cpu()),
                'out': prediction.detach().cpu().numpy(),
                'labels': label.detach().cpu().numpy(),
                'names': names_array,
            }
            self.test_step_outputs.append(cpu_output_dict)

        return output_dict
    
    def on_test_epoch_end(self, outputs=None):
        # Access the outputs through the dataloader_outputs attribute
        if not hasattr(self, "test_step_outputs") or not self.test_step_outputs:
            print("No outputs collected from test_step")
            self.test_results = {"out": None, "labels": None}
            return
        
        # Process the collected outputs
        all_out = np.concatenate([o["out"] for o in self.test_step_outputs])
        all_labels = np.concatenate([o["labels"] for o in self.test_step_outputs])
        all_names = None
        if self.test_step_outputs[0]["names"] is not None:
            all_names = np.concatenate([o["names"] for o in self.test_step_outputs])

        results = {"out": all_out, "labels": all_labels, 'names': all_names}

        if 'out_rmsd' in self.test_step_outputs[0]:
            results['out_rmsd'] = np.concatenate([o['out_rmsd'] for o in self.test_step_outputs])
        if 'labels_rmsd' in self.test_step_outputs[0]:
            results['labels_rmsd'] = np.concatenate([o['labels_rmsd'] for o in self.test_step_outputs])
        
        # Save for later use
        self.test_results = results
        
        # Clear the outputs to free memory
        self.test_step_outputs = []

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
    def configure_optimizers(self):
        # Configure a lr discount for the GNN layers to ensure stable training
        gnn_params = []
        other_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                # Check if parameter belongs to GNN layers
                if any(gnn_layer in name for gnn_layer in ['gcn', 'gat', 'gin', 'sage', 'conv', 'gnn', 'egcl']):
                    gnn_params.append(param)
                else:
                    other_params.append(param)

        param_groups = []
        if gnn_params:
            param_groups.append({
                'params': gnn_params,
                'lr': self.hparams.lr * 0.5,
                'name': 'gnn_layers'
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.hparams.lr * 1,
                'name': 'other_layers'
            })

        # Adam optimizer, lr from hparams, no weight decay by default
        optimizer = torch.optim.Adam(
            param_groups
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.95, patience=4, min_lr=1e-6
        )
        # ALTN: Custom scheduler with warm-up and cosine annealing
        warmup_epochs = getattr(self.hparams, 'warmup_epochs', 10)
        max_epochs = getattr(self.hparams, 'max_epochs', 100)
        cycle_length = getattr(self.hparams, 'cycle_length', 20)
        
        def combined_lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warm-up phase
                return epoch / warmup_epochs
            else:
                # After warm-up: combine decay with cyclic behavior
                progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                
                # Exponential decay component
                decay_factor = 0.9 ** ((epoch - warmup_epochs) // 10)
                
                # Cyclic component (triangular wave)
                cycle_progress = ((epoch - warmup_epochs) % cycle_length) / cycle_length
                cyclic_factor = 1.0 + 0.5 * np.sin(2 * np.pi * cycle_progress)
                
                return decay_factor * cyclic_factor
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, combined_lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'bce_with_logits':
            self.loss_function = F.binary_cross_entropy_with_logits
        elif loss == 'bce':
            # self.loss_function = F.binary_cross_entropy
            self.loss_function = F.binary_cross_entropy_with_logits
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
        elif loss == 'mse':
            self.loss_function = F.mse_loss
        else:
            raise ValueError(f"Invalid loss type: {loss}")

    def load_model(self):
        name = 'molas'
        cap_name = getattr(self.hparams, 'model', 'MolAS')
        
        print(f'Name and Camel Name: {name}, {cap_name}')
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), cap_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{cap_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        # Get model identifiers from hparams or defaults
        num_node_features = getattr(self.hparams, 'num_node_features', 960)
        num_classes = getattr(self.hparams, 'num_classes')
        dropout_rate = getattr(self.hparams, 'dropout_rate', 0.3)

        # Prepare arguments for the main model
        args1 = {
            'num_node_features': num_node_features,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate
        }
        args1.update(other_args)
        return Model(**args1)

def compute_rank_loss(loss_function, predictions, labels, query_lengths):
    loss = loss_function(
        scores=predictions,
        relevance=labels,
        n=query_lengths
    )
    # For the cases where batch size > 1
    if loss.numel() > 1:
        loss = loss.mean()
    return loss

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
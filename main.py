# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args, plot_rmsd_metrics, refine_portfolio

from data.dataset import prepare_data_composite, prepare_data_pose, prepare_data_rmsd
from sklearn.model_selection import train_test_split,  KFold

from utils import ndcg_score
from tqdm import tqdm

import os
# Optimize for RTX 4090 Tensor Cores and performance  
# torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True  # Enable for better performance
torch.backends.cudnn.deterministic = False  # Allow for better performance

import math

import torch.multiprocessing as mp

def load_callbacks(args):
    callbacks = []
    
    # Early stop when validation accuracy fails to increase for 30 validation checks
    callbacks.append(plc.EarlyStopping(
        monitor='val_as_acc',
        mode='max',
        patience=50,
        min_delta=0.001
    ))

    # Options: val_loss(min), val_as_acc(max)
    callbacks.append(plc.ModelCheckpoint(
        monitor='val_as_acc',
        filename='best-{epoch:03d}-{val_loss:.4f}-{val_as_acc:.4f}',
        save_top_k=3,
        mode='max',
        save_last=True
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='low-val-loss-{epoch:03d}-{val_loss:.4f}-{val_as_acc:.4f}',
        save_top_k=3,
        mode='min',
        save_last=False
    ))

    return callbacks

# Main function for training 
def train(args):
    pl.seed_everything(args.seed) # Fix seed for reproducibility
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    
    # Check number of data samples
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print('Number of data samples in train set: ', len(train_loader.dataset))
    print('Number of data samples in val set: ', len(val_loader.dataset))
    print('Number of data samples in test set: ', len(test_loader.dataset))

    # # Filter mask by similarity
    # val_dataset = data_module.valset.reset_index(drop=True)
    # # Extract the values to a numpy array
    # val_data = val_dataset.iloc[:, 2:-1].values

    # chosen_idx = refine_portfolio(val_data, cut_height=0.4, method="cluster")
    # # chosen_idx = refine_portfolio(val_data, method="facility", K=5)
    # mask = np.zeros(args.num_classes, dtype=bool)
    # mask[chosen_idx] = True
    # chosen_methods = val_dataset.columns[2:-1][mask]

    # print(f'Chosen models after refinement: {chosen_methods}')
    # print(mask.astype(int))

    deg = data_module.get_train_deg()
    print("Max node degree computed")

    if load_path is None:
        model = MInterface(deg=deg, **vars(args))
    else:
        model = MInterface(deg=deg, **vars(args))
        args.ckpt_path = load_path

    callbacks = load_callbacks(args)
    logger = TensorBoardLogger(
        save_dir=args.log_root,
        name=args.log_name,
        version=args.log_version
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=1, 
        strategy='ddp_find_unused_parameters_true'
    )
    trainer.fit(model, data_module)

    # np.save(args.ckpt_path[:-5]+'_chosen_models.npy', mask.astype(int))

def test(args):
    pl.seed_everything(args.seed) # Fix seed for reproducibility
    # Load model and data
    data_module = DInterface(**vars(args))
    data_module.setup()

    # # Filter mask by similarity
    # val_dataset = data_module.valset.reset_index(drop=True)
    # # Extract the values to a numpy array
    # val_data = val_dataset.iloc[:, 2:-1].values

    # chosen_idx = refine_portfolio(val_data, cut_height=0.4, method="cluster")
    # # chosen_idx = refine_portfolio(val_data, method="facility", K=24)
    # mask = np.zeros(args.num_classes, dtype=bool)
    # mask[chosen_idx] = True
    # chosen_methods = val_dataset.columns[2:-1][mask]
    # mask = mask.astype(int)
    # print(f'Number of chosen models after refinement: {len(chosen_methods)}')

    test_loader = data_module.test_dataloader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MInterface.load_from_checkpoint(args.ckpt_path, map_location=device, **vars(args))
    model = model.eval()

    if args.test_eval_time:
        import time
        times = []

        # GPU warm-up
        with torch.no_grad():
            for i, batch_example in enumerate(test_loader):
                ligand_data, protein_data, label = batch_example
                
                _ = model(ligand_data.to(device), protein_data.to(device))
                torch.cuda.synchronize()
                if i > 10:
                    break

        # Actual timing
        with torch.no_grad():
            for batch in test_loader:
                batch = batch
                ligand_data, protein_data, label = batch

                start = time.time()
                _ = model(ligand_data.to(device), protein_data.to(device))
                torch.cuda.synchronize()
                end = time.time()

                times.append(end - start)

        avg_time = sum(times) / len(times)
        print(f"Average inference time per batch: {avg_time:.6f} s")
        print(f"Average per-sample time: {avg_time / args.batch_size:.6f} s")
        return

    if args.get_embedded_features:
        # Extract embedded features for all test samples
        all_features = []
        all_names = []
        with torch.no_grad():
            for ligand_data, protein_data, _ in tqdm(test_loader, desc="Extracting Features"):
                ligand_data = ligand_data.to(device)
                protein_data = protein_data.to(device)
                features = model.forward(ligand_data, protein_data, return_features=True)
                all_features.append(features.cpu().numpy())
                all_names.extend(protein_data.names)  # Assuming protein_data has a 'names' attribute

        all_features = np.vstack(all_features)
        # Save to .pt
        torch.save({'names': all_names, 'features': all_features}, args.ckpt_path[:-5]+'_embedded_features.pt')
        return

    num_classes = getattr(args, 'num_classes')

    print('Available devices: ', args.devices[0])
    trainer = Trainer(accelerator="cuda", logger=False, enable_checkpointing=False, devices=[args.devices[0]])

    results = trainer.test(model, datamodule=data_module)
    
    print('Number of data samples in test set: ', len(test_loader.dataset))
    print(results)
    
    # Prepare dataset
    if isinstance(args.benchmark, list):
        
        for b in args.benchmark:
            ds, _ = prepare_data_composite(args.incl_columns, b)
            pose_ds, _ = prepare_data_pose(args.incl_columns, b)
            rmsd_ds, _ = prepare_data_rmsd(args.incl_columns, b)

            if 'dataset' in locals():
                dataset = pd.concat([dataset, ds], ignore_index=True)
                pose_dataset = pd.concat([pose_dataset, pose_ds], ignore_index=True)
                rmsd_dataset = pd.concat([rmsd_dataset, rmsd_ds], ignore_index=True)
            else:
                dataset = ds
                pose_dataset = pose_ds
                rmsd_dataset = rmsd_ds
            
    elif isinstance(args.benchmark, str):
        dataset, _ = prepare_data_composite(args.incl_columns, args.benchmark)
    else:
        raise ValueError("benchmark must be a string or a list of strings")

    if args.test_size == 1:
        test_df = dataset
        test_df_pose = pose_dataset
        test_df_rmsd = rmsd_dataset
    else:
        #!!!!!!!!!!!!!!! Caution: hard-coded here!!!!!!!!!!!!!!!!
        dataset, _ = prepare_data_composite(args.incl_columns, args.benchmark[0])
        pose_dataset, _ = prepare_data_pose(args.incl_columns, args.benchmark[0])
        rmsd_dataset, _ = prepare_data_rmsd(args.incl_columns, args.benchmark[0])

        if args.k_folds is not None:
            # K-Fold cross-validation
            kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
            folds = list(kf.split(dataset))
            train_idx, val_idx = folds[args.fold_num]
            
            train_val = dataset.iloc[train_idx]
            test_df = dataset.iloc[val_idx]
            
            train_val_pose = pose_dataset.iloc[train_idx]
            test_df_pose = pose_dataset.iloc[val_idx]
            
            train_val_rmsd = rmsd_dataset.iloc[train_idx]
            test_df_rmsd = rmsd_dataset.iloc[val_idx]

        else:
            # Split the raw data for testing
            train_val, test_df = train_test_split(dataset, test_size=args.test_size, random_state=args.seed)
            # train_df, val_df = train_test_split(train_val, test_size=args.test_size, random_state=args.seed)
            
            # Split the raw pose and RMSD data for testing as well
            train_val_pose, test_df_pose = train_test_split(pose_dataset, test_size=args.test_size, random_state=args.seed)
            # train_df_pose, val_df_pose = train_test_split(train_val_pose, test_size=args.test_size, random_state=args.seed)
            
            train_val_rmsd, test_df_rmsd = train_test_split(rmsd_dataset, test_size=args.test_size, random_state=args.seed)
            # train_df_rmsd, val_df_rmsd = train_test_split(train_val_rmsd, test_size=args.test_size, random_state=args.seed)

    print('First tested PDB_CCD_ID: ', test_df.iloc[0,0] + '_' + test_df.iloc[0,1])
    print('Last tested PDB_CCD_ID: ', test_df.iloc[-1,0] + '_' + test_df.iloc[-1,1])

    out = model.test_results["out"]
    labels = model.test_results["labels"]
    names = model.test_results["names"]

    test_df = test_df.reset_index(drop=True)  # Reset index to avoid KeyError
    choices = test_df.columns[2:-1]
    cap_res_arr = out.reshape(-1, num_classes)  # Ensure shape is (-1, num_classes)
    print(choices, cap_res_arr.shape)
    cap_df = pd.DataFrame(cap_res_arr, columns=choices) #.rank(axis=1, method='min')
    name_df = pd.DataFrame(names, columns=['Ligand'])
    cap_out = pd.concat([name_df, cap_df], axis=1)
    # Identify selected model for each protein-ligand pair
    cap_out['selected_model'] = cap_out.iloc[:,1:].idxmax(axis=1)
    # print(cap_out)

    # Save the predicted scores
    cap_out.to_csv(args.ckpt_path[:-5]+'_predicted_scores.csv', index=False)

    descriptive = test_df.iloc[:,:-1]
    best_method = []
    # print(descriptive)

    # Create descriptive_pose and descriptive_rmsd dataframes
    descriptive_pose = test_df_pose.iloc[:,:-1]
    descriptive_rmsd = test_df_rmsd.iloc[:,:-1]
    
    # Reset indices to ensure alignment
    descriptive = descriptive.reset_index(drop=True)
    descriptive_pose = descriptive_pose.reset_index(drop=True)
    descriptive_rmsd = descriptive_rmsd.reset_index(drop=True)

    for i in range(len(descriptive)):
        temp_name = descriptive.protein[i]+'_'+descriptive.ligand[i]
        method = cap_out[cap_out.Ligand == temp_name].selected_model.values[0]
        best_method.append(method)

    # SBS over entire benchmark
    df = dataset.reset_index(drop=True).iloc[:,:-1]
    means = df.iloc[:,2:].mean(axis=0)
    sbs = means.idxmax()
    # print(f'SBS over entire benchmark: {sbs} with mean score {means[sbs]}')
        
    descriptive[f'SBS: {sbs}'] = descriptive[sbs]
    descriptive_pose[f'SBS: {sbs}'] = descriptive_pose[sbs]
    descriptive_rmsd[f'SBS: {sbs}'] = descriptive_rmsd[sbs]

    # Apply same operations to all three dataframes
    descriptive['choose'] = best_method
    descriptive_pose['choose'] = best_method
    descriptive_rmsd['choose'] = best_method
    
    best_method_oracle = descriptive.iloc[:,2:-1].idxmax(axis=1)
        
    descriptive['oracle'] = descriptive.iloc[:,2:-1].max(axis=1)
    descriptive_pose['oracle'] = descriptive_pose.iloc[:,2:-1].max(axis=1)
    descriptive_rmsd['oracle'] = descriptive_rmsd.iloc[:,2:-1].min(axis=1)  # For RMSD, best is minimum
    
    # Ensure model_scores are numeric and compute 2nd and 3rd best for scoring data
    model_scores = descriptive.iloc[:,2:-1].apply(pd.to_numeric, errors='coerce')
    descriptive['2nd'] = model_scores.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    descriptive['3rd'] = model_scores.apply(lambda row: row.nlargest(3).iloc[-1], axis=1)
    
    # For pose data (binary/boolean), compute 2nd and 3rd differently
    model_scores_pose = descriptive_pose.iloc[:,2:-1].apply(pd.to_numeric, errors='coerce')
    # Check if pose data is binary (0/1) or continuous
    if descriptive_pose.iloc[:,2:-1].dtypes.apply(lambda x: x in ['bool', 'int64', 'float64']).all():
        # For binary pose data, 2nd and 3rd largest make sense
        descriptive_pose['2nd'] = model_scores_pose.apply(lambda row: row.nlargest(2).iloc[-1] if len(row.dropna()) >= 2 else np.nan, axis=1)
        descriptive_pose['3rd'] = model_scores_pose.apply(lambda row: row.nlargest(3).iloc[-1] if len(row.dropna()) >= 3 else np.nan, axis=1)
    else:
        # For non-numeric pose data, set to NaN
        descriptive_pose['2nd'] = np.nan
        descriptive_pose['3rd'] = np.nan
    
    # For RMSD data, use nsmallest since lower RMSD is better
    model_scores_rmsd = descriptive_rmsd.iloc[:,2:-1].apply(pd.to_numeric, errors='coerce')
    descriptive_rmsd['2nd'] = model_scores_rmsd.apply(lambda row: row.nsmallest(2).iloc[-1] if len(row.dropna()) >= 2 else np.nan, axis=1)
    descriptive_rmsd['3rd'] = model_scores_rmsd.apply(lambda row: row.nsmallest(3).iloc[-1] if len(row.dropna()) >= 3 else np.nan, axis=1)
    
    # Apply AS calculation to all three dataframes
    values = descriptive.apply(lambda row: row[row['choose']], axis=1)
    descriptive.insert(args.num_classes+2, 'AS', values)
    
    values_pose = descriptive_pose.apply(lambda row: row[row['choose']], axis=1)
    descriptive_pose.insert(args.num_classes+2, 'AS', values_pose)
    
    values_rmsd = descriptive_rmsd.apply(lambda row: row[row['choose']], axis=1)
    descriptive_rmsd.insert(args.num_classes+2, 'AS', values_rmsd)
    
    # Now we have three complete dataframes, use them directly for evaluation
    scoring_columns = [f'SBS: {sbs}', 'AS', 'oracle']
    # scoring_columns = [f'SBS: {sbs}', 'AS', 'oracle', '2nd', '3rd']
    # scoring_columns = ['surf', 'unimol','interformer', 'autodock', 'gnina', 'alphafold3', 'boltz1x', 'sbs', 'AS', 'oracle', '2nd', '3rd']
    
    # Detailed metrics: RMSD < 1; RMSD < 2; Med RMSD; RMSD < 2 & Pose valid
    eval_results = dict()
    for col in scoring_columns:
        if col in descriptive.columns:
            # Extract values from respective dataframes
            rmsd_values = descriptive_rmsd[col]
            pose_values = descriptive_pose[col]
            
            # 1. Percent of RMSD < 1
            percent_rmsd_lt_1 = (rmsd_values < 1).mean() * 100
            # 2. Percent of RMSD < 2
            percent_rmsd_lt_2 = (rmsd_values < 2).mean() * 100                    
            # 3. Median RMSD
            median_rmsd = rmsd_values.median()
            # 4. Percent of RMSD < 2 & Pose valid
            percent_rmsd_lt_2_valid = ((rmsd_values < 2) & pose_values.astype(bool)).mean() * 100
            # 5. Percent of RMSD < 5
            percent_rmsd_lt_5 = (rmsd_values < 5).mean() * 100   
            # 6. Percent of RMSD < 1 & Pose valid
            percent_rmsd_lt_1_valid = ((rmsd_values < 1) & pose_values.astype(bool)).mean() * 100

            eval_results[col] = {
                'Percent RMSD < 1': round(percent_rmsd_lt_1, 2),
                'Percent RMSD < 2': round(percent_rmsd_lt_2, 2),
                'Percent RMSD < 5': round(percent_rmsd_lt_5, 2),
                'Median RMSD': round(median_rmsd, 3),
                'Percent RMSD < 1 (Valid Poses)': round(percent_rmsd_lt_1_valid, 2) if percent_rmsd_lt_1_valid is not None else 'N/A',
                'Percent RMSD < 2 (Valid Poses)': round(percent_rmsd_lt_2_valid, 2) if percent_rmsd_lt_2_valid is not None else 'N/A',
            }

    # print(descriptive)
    print(descriptive.iloc[:,1:].sum(axis=0))

    # Count occurrences of each algorithm in the 'choose' column (same for all dataframes)
    def count_selected_algorithms(descriptive_df):
        """
        Count the number of times each algorithm was selected.
        Args:
            descriptive_df: DataFrame containing the 'choose' column
        Returns:
            pandas.Series: Count of each algorithm selection
        """
        algorithm_counts = descriptive_df['choose'].value_counts()
        return algorithm_counts

    # Call the function and display results
    algorithm_counts = count_selected_algorithms(descriptive)
    print("\nAlgorithm Selection Counts:")
    print(algorithm_counts)
    # Optional: Display as percentages
    # algorithm_percentages = (algorithm_counts / len(descriptive)) * 100
    # print("\nAlgorithm Selection Percentages:")
    # for algo, percentage in algorithm_percentages.items():
    #     print(f"{algo}: {percentage:.2f}%")
        
    oracle_counts = best_method_oracle.value_counts()
    print("\nOracle Selection Counts:")
    print(oracle_counts)
    # oracle_percentages = (oracle_counts / len(descriptive)) * 100
    # print("\nOracle Selection Percentages:")
    # for algo, percentage in oracle_percentages.items():
    #     print(f"{algo}: {percentage:.2f}%")

    # display evalutation results
    print("\nEvaluation Results:")
    for col, metrics in eval_results.items():
        print(f"\n{col}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    plot_rmsd_metrics(eval_results)

    full_eval_results = dict()
    for col in descriptive.columns:
        if col not in ['protein', 'ligand', 'choose']:
            # Extract values from respective dataframes
            rmsd_values = descriptive_rmsd[col]
            pose_values = descriptive_pose[col]
            
            # 1. Percent of RMSD < 1
            percent_rmsd_lt_1 = (rmsd_values < 1).mean() * 100
            # 2. Percent of RMSD < 2
            percent_rmsd_lt_2 = (rmsd_values < 2).mean() * 100                    
            # 3. Median RMSD
            median_rmsd = rmsd_values.median()
            # 4. Percent of RMSD < 2 & Pose valid
            percent_rmsd_lt_2_valid = ((rmsd_values < 2) & pose_values.astype(bool)).mean() * 100
            # 5. Percent of RMSD < 5
            percent_rmsd_lt_5 = (rmsd_values < 5).mean() * 100   
            # 6. Percent of RMSD < 1 & Pose valid
            percent_rmsd_lt_1_valid = ((rmsd_values < 1) & pose_values.astype(bool)).mean() * 100

            full_eval_results[col] = {
                'Percent RMSD < 1': round(percent_rmsd_lt_1, 2),
                'Percent RMSD < 2': round(percent_rmsd_lt_2, 2),
                'Percent RMSD < 5': round(percent_rmsd_lt_5, 2),
                'Median RMSD': round(median_rmsd, 3),
                'Percent RMSD < 1 (Valid Poses)': round(percent_rmsd_lt_1_valid, 2) if percent_rmsd_lt_1_valid is not None else 'N/A',
                'Percent RMSD < 2 (Valid Poses)': round(percent_rmsd_lt_2_valid, 2) if percent_rmsd_lt_2_valid is not None else 'N/A',
            }
    
    oracle_sbs_gap_closed = {}
    
    for metric in eval_results['AS'].keys():
        total_gap = eval_results['oracle'][metric] - eval_results['SBS: '+sbs][metric]
        closed_gap = eval_results['AS'][metric] - eval_results['SBS: '+sbs][metric]
        oracle_sbs_gap_closed[metric] = (closed_gap / total_gap * 100) if total_gap !=0 else 0.0

    eval_results['VBS_SBS_Gap_Closed_by_AS(%)'] = oracle_sbs_gap_closed
        
    print("\nOracle-SBS Gap Closed by AS (%):")
    for metric, value in oracle_sbs_gap_closed.items():
        print(f"{metric}: {value:.2f}%")

    AS_rmsd_values = descriptive_rmsd['AS']
    AS_pose_values = descriptive_pose['AS']
    rmsd_2_valid = ((AS_rmsd_values < 2) & AS_pose_values.astype(bool)).values.astype(int)
    rmsd_1_valid = ((AS_rmsd_values < 1) & AS_pose_values.astype(bool)).values.astype(int)

    df_algorithm_counts = pd.DataFrame({
        'Algorithm': list(descriptive.columns[2:-6]),
        'Selection_Count': np.zeros(len(descriptive.columns[2:-6]), dtype=int),
        'Oracle_Count': np.zeros(len(descriptive.columns[2:-6]), dtype=int)
    })
    for idx, algo in enumerate(descriptive.columns[2:-6]):
        if algo in algorithm_counts.index:
            df_algorithm_counts.at[idx, 'Selection_Count'] = int(algorithm_counts[algo])
        if algo in oracle_counts.index:
            df_algorithm_counts.at[idx, 'Oracle_Count'] = int(oracle_counts[algo])
    
    descriptive['oracle_choose'] = best_method_oracle

    # Save results to npy
    np.save(args.ckpt_path[:-5]+'_1_threshold.npy', rmsd_1_valid)
    np.save(args.ckpt_path[:-5]+'_2_threshold.npy', rmsd_2_valid)
    pd.DataFrame(eval_results).to_csv(args.ckpt_path[:-5]+'.csv')
    pd.DataFrame(full_eval_results).to_csv(args.ckpt_path[:-5]+'_full.csv')
    descriptive.to_csv(args.ckpt_path[:-5]+'_descriptive.csv')
    # Save algorithm counts and oracle counts
    df_algorithm_counts.to_csv(args.ckpt_path[:-5]+'_algorithm_selection_counts.csv', index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    # LR Scheduler (not used in notebook, set to None)
    parser.add_argument('--lr_scheduler', default=None, type=str)
    parser.add_argument('--weight_decay', default=0, type=float)

    # Model and Data
    parser.add_argument('--model', default='MolASNoGNN', type=str)
    parser.add_argument('--loss', default='bce_with_logits', type=str)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    
    ## Data related
    parser.add_argument('--num_node_features', default=960, type=int)
    parser.add_argument('--incl_columns', default=[], type=str, nargs='*', help='Columns to include (space-separated list)')
    parser.add_argument('--num_classes', default=24, type=int)  # This need to be changed with respect to the columns(algorithms) dropped
    parser.add_argument('--test_size', default=0.1, type=float)  # Test size for split

    ## Dataset Configuration
    parser.add_argument('--benchmark', default='posex_self_docking', type=str, nargs='*', help='Benchmark dataset to use')

    ## KFold Support
    parser.add_argument('--k_folds', default=None, type=int)
    parser.add_argument('--fold_num', default=0, type=int)
    ## nDCG K config (eval)
    parser.add_argument('--ndcg_k', default=3, type=int, help='Rank cutoff for nDCG calculation')

    # Trainer args
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--accelerator', default='cuda', type=str)
    parser.add_argument('--devices', default=[0], type=int, nargs='+')
    parser.add_argument('--log_root', default='lightning_logs', type=str, help='Root directory for TensorBoard log files')
    parser.add_argument('--log_name', default='default', type=str, help='Experiment subdirectory name for logging')
    parser.add_argument('--log_version', default=None, type=str, help='Optional version identifier for the logger (auto if omitted)')

    # Test mode
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--ckpt_path', default=None, type=str, help='Path to the model checkpoint for testing')

    # Additional "rank" model parameters
    parser.add_argument('--use_ndcg_loss', action='store_true', default=False, help='Enable NDCG loss component')
    parser.add_argument('--no_ndcg_loss', dest='use_ndcg_loss', action='store_false', help='Disable NDCG loss component')
    parser.add_argument('--ndcg_loss_weight', type=float, default=1.0, help='Weight for NDCG loss component')
    parser.add_argument('--sigma_ndcg', type=float, default=1.0)
    
    parser.add_argument('--use_logistic_loss', action='store_true', default=False, help='Enable logistic loss component')
    parser.add_argument('--no_logistic_loss', dest='use_logistic_loss', action='store_false', help='Disable logistic loss component')
    parser.add_argument('--logistic_loss_weight', type=float, default=0.01, help='Weight for logistic loss component')
    parser.add_argument('--sigma_logistic', type=float, default=1.0)

    parser.add_argument('--get_embedded_features', action='store_true', help='Extract embedded features from the model')
    parser.add_argument('--test_eval_time', action='store_true', help='Evaluate inference time during testing')

    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    
    torch.autograd.set_detect_anomaly(True)
    
    if args.test:
        test(args)
    else:
        train(args)
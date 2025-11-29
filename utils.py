# Copyright 2021 Zhongyang Zhang
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

import os
from pathlib2 import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster

def refine_portfolio(M_train, eps=0.0, cut_height=0.5, method="cluster", K=5):
    """
    M_train: (n_instances, n_algorithms) training composite scores
    method:  "cluster"  -> rank-correlation clustering + rep by coverage
             "facility" -> greedy facility-location selection
    Returns: list of selected algorithm indices
    """

    # --- normalize per instance (ranks to remove scale) ---
    ranks = np.apply_along_axis(lambda r: r.argsort().argsort(), 1, M_train)
    M_norm = ranks / ranks.max(axis=1, keepdims=True)

    if method == "facility":
        # Greedy facility-location: pick algs that cover most instances
        n, A = M_norm.shape
        chosen, current = [], np.full(n, -np.inf)
        for _ in range(K):
            gains = [np.maximum(current, M_norm[:,a]).sum() - current.sum()
                     for a in range(A) if a not in chosen]
            a = [a for a in range(A) if a not in chosen][int(np.argmax(gains))]
            chosen.append(a)
            current = np.maximum(current, M_norm[:,a])
        return chosen

    else:
        # --- Spearman correlation between algorithms ---
        corr, _ = spearmanr(M_norm, axis=0)
        dist = (1 - corr) / 2
        iu = np.triu_indices(dist.shape[0], k=1)
        Z = linkage(dist[iu], method='average')
        labels = fcluster(Z, t=cut_height, criterion='distance')

        # --- pick one rep per cluster (best mean score) ---
        reps = []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            mean_scores = M_train[:, idx].mean(axis=0)
            reps.append(idx[np.argmax(mean_scores)])
        return reps

def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root==version==v_num==None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res

def load_model_path_by_args(args):
    # Use getattr with default None to avoid AttributeError if missing
    return load_model_path(
        root=getattr(args, 'load_dir', None),
        version=getattr(args, 'load_ver', None),
        v_num=getattr(args, 'load_v_num', None)
    )

def ndcg_score(y_true, y_score, k=3):
    """
    Compute nDCG for a single sample.
    y_true: true relevance scores (1D tensor or array)
    y_score: predicted scores (1D tensor or array)
    k: rank cutoff (if None, use all)
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    n = len(y_true)
    if n == 0:
        return 0.0
    if k is None or k > n:
        k = n

    # DCG
    order = np.argsort(y_score)[::-1][:k]
    gains = 2 ** y_true[order] - 1
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)

    # iDCG
    ideal_order = np.argsort(y_true)[::-1][:k]
    ideal_gains = 2 ** y_true[ideal_order] - 1
    ideal_dcg = np.sum(ideal_gains / discounts)

    return float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

def plot_rmsd_metrics(eval_results, metrics_to_plot=None):
    """
    Plot horizontal bar charts for RMSD metrics from eval_results dictionary.
    
    Args:
        eval_results (dict): Dictionary containing evaluation results with RMSD metrics
        metrics_to_plot (list): List of metrics to plot. If None, plots all available metrics.
                               Options: ['rmsd_lt_1', 'rmsd_lt_2', 'rmsd_lt_5', 'rmsd_lt_2_valid', 'median_rmsd']
    """
    # Filter to only include specific models
    target_models = ['AS', 'uni', 'surf', 'gnina', 'smina', 'qvina', 'diffL', 'diff',  'karma', 'oracle', '2nd']
    models = [model for model in target_models if model in eval_results]
    
    # Define available metrics and their properties
    metric_configs = {
        'rmsd_lt_1': {
            'key': 'Percent RMSD < 1',
            'label': 'RMSD < 1',
            'color': 'lightgreen',
            'unit': '%',
            'is_percentage': True
        },
        'rmsd_lt_2': {
            'key': 'Percent RMSD < 2',
            'label': 'RMSD < 2',
            'color': 'skyblue',
            'unit': '%',
            'is_percentage': True
        },
        'rmsd_lt_2_valid': {
            'key': 'Percent RMSD < 2 (Valid Poses)',
            'label': 'RMSD < 2 (Valid poses)',
            'color': 'lightcoral',
            'unit': '%',
            'is_percentage': True
        },
        'rmsd_lt_5': {
            'key': 'Percent RMSD < 5',
            'label': 'RMSD < 5',
            'color': 'orange',
            'unit': '%',
            'is_percentage': True
        },
        'median_rmsd': {
            'key': 'Median RMSD',
            'label': 'Median RMSD',
            'color': 'purple',
            'unit': 'Å',
            'is_percentage': False
        }
    }
    
    # Default to plotting percentage metrics if not specified (now in the desired order)
    if metrics_to_plot is None:
        metrics_to_plot = ['rmsd_lt_1', 'rmsd_lt_2', 'rmsd_lt_2_valid', 'rmsd_lt_5']
    
    # Check if we need separate subplots for different units
    has_percentage = any(metric_configs[m]['is_percentage'] for m in metrics_to_plot)
    has_non_percentage = any(not metric_configs[m]['is_percentage'] for m in metrics_to_plot)
    
    if has_percentage and has_non_percentage:
        # Create subplots for different units
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot percentage metrics
        percentage_metrics = [m for m in metrics_to_plot if metric_configs[m]['is_percentage']]
        _plot_metrics_group(ax1, models, eval_results, percentage_metrics, metric_configs, 'Percentage (%)')
        
        # Plot non-percentage metrics
        non_percentage_metrics = [m for m in metrics_to_plot if not metric_configs[m]['is_percentage']]
        _plot_metrics_group(ax2, models, eval_results, non_percentage_metrics, metric_configs, 'RMSD (Å)')
        
    else:
        # Single subplot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        unit_label = 'Percentage (%)' if has_percentage else 'RMSD (Å)'
        _plot_metrics_group(ax, models, eval_results, metrics_to_plot, metric_configs, unit_label)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def _plot_metrics_group(ax, models, eval_results, metrics_to_plot, metric_configs, xlabel):
    """Helper function to plot a group of metrics on a single axis."""
    # Extract data for all metrics
    metric_data = {}
    for metric in metrics_to_plot:
        config = metric_configs[metric]
        values = []
        for model in models:
            val = eval_results[model].get(config['key'], 0)
            # Handle 'N/A' values
            if val == 'N/A':
                val = 0
            values.append(val)
        metric_data[metric] = values
    
    # Set up bar positions
    y_pos = np.arange(len(models))
    n_metrics = len(metrics_to_plot)
    width = 0.8 / n_metrics  # Adjust width based on number of metrics
    
    # Create horizontal bars
    bars = []
    for i, metric in enumerate(metrics_to_plot):
        config = metric_configs[metric]
        offset = (i - (n_metrics - 1) / 2) * width
        bar = ax.barh(y_pos + offset, metric_data[metric], width,
                     label=config['label'], color=config['color'], alpha=0.7)
        bars.append(bar)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel(xlabel)
    ax.set_title(f'RMSD Metrics Comparison ({xlabel.split("(")[0].strip()})')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar_group in enumerate(bars):
        config = metric_configs[metrics_to_plot[i]]
        for bar in bar_group:
            width_val = bar.get_width()
            if config['is_percentage']:
                label = f'{width_val:.1f}%'
            else:
                label = f'{width_val:.2f}Å'
            ax.text(width_val + (ax.get_xlim()[1] * 0.01), bar.get_y() + bar.get_height()/2, 
                    label, ha='left', va='center', fontsize=8)
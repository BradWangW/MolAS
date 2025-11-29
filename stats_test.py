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
import torch
import pandas as pd
import numpy as np
from argparse import ArgumentParser

torch.backends.cudnn.benchmark = True  # Enable for better performance
torch.backends.cudnn.deterministic = False  # Allow for better performance

import numpy as np
from typing import Dict
from statsmodels.stats.contingency_tables import mcnemar

from tabulate import tabulate

def compare_binary_selectors(
    yA: np.ndarray, yB: np.ndarray, n_boot: int = 10000, seed: int = 0
) -> Dict[str, float]:
    """
    Compare two binary docking success outputs (per-instance) with paired statistics.

    Args:
        yA, yB : arrays of shape (n,) with {0,1} entries (binary success indicators).
        n_boot : number of bootstrap resamples for CI.
        seed   : random seed for reproducibility.

    Returns:
        dict with:
            - mean_A: mean success of A
            - mean_B: mean success of B
            - mean_diff: mean(A-B)
            - ci_low, ci_high: bootstrap 95% CI of mean_diff
            - p_mcnemar: exact McNemar test p-value (two-sided)
            - n: number of paired instances
    """
    yA = np.asarray(yA).astype(float)
    yB = np.asarray(yB).astype(float)
    assert yA.shape == yB.shape

    n = len(yA)
    diff = yA - yB
    mean_A = float(yA.mean())
    mean_B = float(yB.mean())
    mean_diff = float(diff.mean())

    # Bootstrap CI on mean difference
    rng = np.random.default_rng(seed)
    boot_means = [diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # McNemar’s test
    n01 = int(((yA == 0) & (yB == 1)).sum())
    n10 = int(((yA == 1) & (yB == 0)).sum())
    print(n01)
    print(n10)
    table = [[0, n01], [n10, 0]]
    result = mcnemar(table, exact=True)
    pval = result.pvalue

    return {
        "n": n,
        "mean_A": mean_A,
        "mean_B": mean_B,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_mcnemar": pval,
    }

def compare_continuous_selectors(
    yA: np.ndarray, yB: np.ndarray, n_boot: int = 10000, seed: int = 0
) -> Dict[str, float]:
    """
    Compare two continuous docking quality outputs (per-instance) with paired statistics.

    Args:
        yA, yB : arrays of shape (n,) with continuous entries (e.g., RMSD).
        n_boot : number of bootstrap resamples for CI.
        seed   : random seed for reproducibility.

    Returns:
        dict with:
            - mean_A: mean quality of A
            - mean_B: mean quality of B
            - mean_diff: mean(A-B)
            - ci_low, ci_high: bootstrap 95% CI of mean_diff
            - p_mcnemar: exact McNemar test p-value (two-sided)
    """
    yA = np.asarray(yA).astype(float)
    yB = np.asarray(yB).astype(float)
    assert yA.shape == yB.shape

    n = len(yA)
    diff = yA - yB
    mean_A = float(yA.mean())
    mean_B = float(yB.mean())
    mean_diff = float(diff.mean())

    # Bootstrap CI on mean difference
    rng = np.random.default_rng(seed)
    boot_means = [diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # McNemar’s test
    n01 = int(((yA < 0.50124) & (yB > 0.50124)).sum())
    n10 = int(((yA > 0.50124) & (yB < 0.50124)).sum())
    print(n01)
    print(n10)
    table = [[0, n01], [n10, 0]]
    result = mcnemar(table, exact=True)
    pval = result.pvalue

    return {
        "n": n,
        "mean_A": mean_A,
        "mean_B": mean_B,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_mcnemar": pval,
    }

def main(args):
    assert len(args.original_ckpts) == len(args.after_ckpts), "Please provide the same number of original and after finetune checkpoints."

    #--------------------Add-on: averaged results over multiple seeds---------------------#
    eval_results = [pd.read_csv(ckpt[:-5]+'_full.csv') for ckpt in args.after_ckpts]

    eval_results = pd.concat(eval_results).groupby('Unnamed: 0').mean()
    print("Averaged results over multiple seeds:")
    # Recompute the column VBS_SBS_Gap_Closed_by_AS(%)
    # How to extract the second column without using its name? Answer: use iloc
    sbs_col_name = eval_results.columns[-4]
    eval_results['VBS_SBS_Gap_Closed_by_AS(%)'] = (eval_results['AS'] - eval_results[sbs_col_name]) / (eval_results['oracle'] - eval_results[sbs_col_name]) * 100.0
    # Print proper table
    # print(tabulate(eval_results, headers='keys', tablefmt='psql'))
    # Print the transposed version for better readability
    print("\nTransposed view:")
    print(tabulate(eval_results.T, headers='keys', tablefmt='psql'))
    #------------------------------------------------------------------------------------#
    
    algorithm_counts = [pd.read_csv(ckpt[:-5]+'_algorithm_selection_counts.csv') for ckpt in args.after_ckpts]
    algorithm_counts = pd.concat(algorithm_counts).groupby('Algorithm').sum()
    # Divide by column total to get percentages
    algorithm_counts = algorithm_counts.div(algorithm_counts.sum(axis=0), axis=1) * 100.0

    descriptives = [pd.read_csv(ckpt[:-5]+'_descriptive.csv') for ckpt in args.after_ckpts]

    # Merge descriptives
    df_full = pd.concat(descriptives).reset_index(drop=True)
    # Drop first column if it's unnamed or not
    if df_full.columns[0] == 'Unnamed: 0' or not df_full.columns[0]:
        df_full = df_full.drop(df_full.columns[0], axis=1)

    dict_as_vs_sbs = compare_continuous_selectors(df_full[sbs_col_name], df_full["AS"])
    print(df_full[sbs_col_name], df_full["AS"])
    print("Comparison of Algorithm Selection vs SBS:")
    print(dict_as_vs_sbs)

    def precision_when_selected(df):
        oracle = df["oracle_choose"]
        choose = df["choose"]
        precision = []
        for alg in algorithm_counts.index:
            picks = (choose == alg)
            if picks.sum() == 0:
                precision.append(np.nan)
            else:
                precision.append((oracle[picks] == choose[picks]).mean()*100)
        return precision
    
    algorithm_counts['Precision_when_selected(%)'] = precision_when_selected(df_full)
    algorithm_counts['Correctly_chosen(%)'] = (algorithm_counts['Precision_when_selected(%)'] * algorithm_counts['Selection_Count']) / 100.0

    print("\nAveraged Algorithm Selection Counts over ckpts:")
    print(tabulate(algorithm_counts, headers='keys', tablefmt='psql'))
    #------------------------------------------------------------------------------------#

    original_2_results = [np.load(ckpt[:-5]+'_2_threshold.npy') for ckpt in args.original_ckpts]
    original_1_results = [np.load(ckpt[:-5]+'_1_threshold.npy') for ckpt in args.original_ckpts]
    after_2_results = [np.load(ckpt[:-5]+'_2_threshold.npy') for ckpt in args.after_ckpts]
    after_1_results = [np.load(ckpt[:-5]+'_1_threshold.npy') for ckpt in args.after_ckpts]

    print(f'Shapes of original npys: {[arr.shape for arr in original_2_results]}, {[arr.shape for arr in original_1_results]}')
    # Average over seeds
    original_2_results = np.concatenate(original_2_results).flatten()
    original_1_results = np.concatenate(original_1_results).flatten()
    after_2_results = np.concatenate(after_2_results).flatten()
    after_1_results = np.concatenate(after_1_results).flatten()

    dict_2 = compare_binary_selectors(original_2_results, after_2_results)
    dict_1 = compare_binary_selectors(original_1_results, after_1_results)
    print("Comparison of original vs after finetune (RMSD < 2A & valid pose):")
    print(dict_2)
    print("Comparison of original vs after finetune (RMSD < 1A & valid pose):")
    print(dict_1)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--original_ckpts', default=[], type=str, nargs='*', help='Original checkpoints')
    parser.add_argument('--after_ckpts', default=[], type=str, nargs='*', help='After finetune checkpoints')

    args = parser.parse_args()
    
    main(args)
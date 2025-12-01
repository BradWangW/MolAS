# **MolAS: Molecular Embedding–Based Algorithm Selection in Protein–Ligand Docking**

MolAS is a learning-based framework that predicts, for each protein–ligand pair, which docking or pose-generation method is most likely to succeed.
This repository contains the full training, evaluation, and analysis code used in the accompanying paper *Molecular Embedding–Based Algorithm Selection in Protein–Ligand Docking*.

---

## **Repository Layout**

```
molas/
│
├── main.py              # Training / testing entry point (PyTorch Lightning)
├── main.sh              # 5-fold and multi-GPU script (torchrun presets)
│
├── data/
│   ├── dataset.py       # Dataset construction from CSVs and precomputed graphs
│   ├── data_interface.py# LightningDataModule with K-Fold / split support
│   └── ...
│
├── model/               # MolAS architectures and loss functions
│
├── stats_test.py        # Fold-concatenated reporting and paired statistics
├── environment.yaml     # Basic environment (PyTorch installed separately)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## **Installation**

Create a new environment and install PyTorch **separately**.

```bash
# 1) Create env
conda env create -f environment.yaml
conda activate as_dock

# 2) Install PyTorch (choose a CUDA build)
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

# 3) Install remaining dependencies
pip install -r requirements.txt
```

---

## **Data Preparation**

Download the processed MolAS dataset and trained checkpoints from:

**Zenodo: <INSERT_ZENODO_LINK>**

Then update the `data_path` variables in `data/dataset.py`.

### **Expected directory structure**

```
{data_path}/
    ├── protein_graphs_esmc_600m/
    │     └── pyg_graph_{pdbid}_{ligandid}_esmc_600m.pt
    │
    ├── ligand_chemberta.pt
    │
    ├── {benchmark}_pb_ratio.csv     # PoseBusters validity (per-algorithm)
    └── {benchmark}_rmsd.csv         # RMSD targets (per-algorithm)
```

### Notes on checkpoints

Log folder of checkpoints (named *lightning_logs*) should be placed inside the repository root, **parallel** to this README.md.

---

## **Quick Start**

### **5-Fold and Multi-GPU Training**

Use the preset runner:

```bash
bash main.sh
```

Configure:

* `benchmark` (e.g., `posex_self_docking`, `posex_cross_docking`, `astex_posex`, `moad`, `posebusters`)
* `relaxation` (`both | true | false`)
* `incl_columns` (algorithm list; ensure count matches `--num_classes`)
* `devices` (e.g., `"0,1,2,3"` for torchrun)
* `test=1` to run test-only mode and export reports
* `version=0` to set the initial version of the 5-version bundle for testing

#### **Version Directory**

To reproduce the results using the Zenodo-released checkpoints, set `version` according to the target benchmark and post-processing mode. Each entry refers to a contiguous 5-checkpoint bundle (`version`–`version+4`).

| Benchmark     | Post-process        | `version` | Version bundle |
| ------------- | ------------------- | --------- | -------------- |
| MOAD-curated  | —                   | 5         | 5–9            |
| PoseX + Astex | Mixed               | 219       | 219–223        |
| PoseX-SD      | Mixed               | 5         | 5–9            |
| PoseX-CD      | Mixed               | 5         | 5–9            |
| PoseBusters   | Mixed               | 20        | 20–24          |
| PoseX + Astex | No                  | 209       | 209–213        |
| PoseX + Astex | Relaxation          | 214       | 214–218        |
| PoseX-SD      | No                  | 10        | 10–14          |
| PoseX-SD      | Relaxation          | 30        | 30–34          |
| PoseX-CD      | No                  | 30        | 30–34          |
| PoseX-CD      | Relaxation          | 25        | 25–29          |
| PoseBusters   | No                  | 10        | 10–14          |
| PoseBusters   | Energy minimisation | 15        | 15–19          |


### **Single-GPU Example**

```bash
python main.py \
    --benchmark posex_self_docking \
    --batch_size 16 \
    --lr 1e-3 \
    --max_epochs 100 \
    --devices 0 \
    --model MolAS \
    --num_node_features 1152 \
    --num_classes 24 \
    --incl_columns surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer
```

### **Testing With a Checkpoint**

```bash
python main.py \
    --benchmark posex_self_docking \
    --devices 0 \
    --model MolAS \
    --num_node_features 1152 \
    --num_classes 24 \
    --incl_columns <algorithm list> \
    --test \
    --ckpt_path path/to/checkpoint.ckpt
```

---

## **Important Arguments**

* `--model`
  `MolAS` (default) or multi-GNN variants (e.g., `MolAS_GCN_GAT_GINE`, `MolAS_EGNN_GAT_GINE`, `MolASGT`).
  Multi-GNN variants require node features and adjacency consistent with provided graphs.

* `--incl_columns`
  Space-separated algorithm names. Must match `--num_classes`.

* `--use_ndcg_loss`, `--ndcg_loss_weight`, `--ndcg_k`
  Enable LambdaNDCG ranking loss.

* `--use_logistic_loss`, `--logistic_loss_weight`
  Pairwise logistic ranking objective.

* `--k_folds`, `--fold_num`
  Enable K-fold cross-validation.

* `--log_root`, `--log_name`, `--log_version`
  Control Lightning logging layout.

---

## **License**

Apache License 2.0.

---

## **Acknowledgements**

MolAS uses embeddings from **ESM-C** and **ChemBERTa**, and builds on PyTorch, PyTorch Lightning, and PyTorch Geometric.

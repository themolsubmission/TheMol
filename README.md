<p align="center">
  <img src="assets/logo.png" width="300" alt="TheMol Logo"/>
</p>

<h1 align="center">Learning Canonical Representations for Unified 3D Molecular Modeling</h1>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg" alt="arXiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python"></a>
</p>

<p align="center">
  📄 <a href="https://arxiv.org/abs/xxxx.xxxxx">Paper</a> •
  🧠 <a href="https://drive.google.com/drive/folders/YOUR_MODEL_LINK">Model</a> •
  🗂️ <a href="https://drive.google.com/drive/folders/YOUR_DATA_LINK">Dataset</a> •
  💻 <a href="#quick-start">Quick Start</a>
</p>

---

## 📋 Overview

3D molecular foundation models must handle diverse tasks (from predicting scalar properties to generating 3D coordinates) yet existing approaches force a choice between invariant and equivariant architectures, each with inherent limitations.

**We show that this tradeoff is unnecessary.**

By canonicalizing molecules into a learned standard pose before encoding, a single non-equivariant model can support both invariant and equivariant tasks. We theoretically demonstrate that this approach overcomes the fundamental constraints of prior paradigms. Pretrained on large-scale molecular data, our model consistently rivals methods purpose-built for each task.

<p align="center">
  <img src="assets/framework.png" width="800" alt="TheMol Framework"/>
</p>

### Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Unified Architecture** | Single model handles both invariant and equivariant tasks |
| 🧬 **Canonical Representation** | Learned standard pose eliminates architecture tradeoffs |
| 🔬 **Multi-task Capability** | Property prediction, generation, and structure-based optimization |
| ⚡ **State-of-the-art Performance** | Rivals purpose-built methods across all benchmarks |

---

## 🔥 News

- **[2025.xx]** Code released for ICML 2025 submission
- **[2025.xx]** Pretrained checkpoints available on Google Drive

---

## 📁 Repository Structure

```
TheMol/
├── unimol/                     # Core library
│   ├── models/                 # Model architectures
│   ├── tasks/                  # Training tasks
│   ├── losses/                 # Loss functions
│   ├── data/                   # Data processing
│   └── utils/                  # Utilities
├── generation/                 # Molecule generation
│   ├── evaluate.py             # Evaluation metrics
│   ├── train_geom.sh           # Train on GEOM dataset
│   ├── train_midi.sh           # Train on MIDI dataset
│   └── train_midi_flow.sh      # Train with flow matching
├── property_pred/              # MoleculeNet property prediction
│   ├── train.sh                # Grid search training
│   └── summarize.sh            # Summarize results
├── admet_pred/                 # TDC ADMET prediction
│   ├── train.sh                # Grid search training
│   └── summarize.sh            # Summarize results
├── optimization/               # Target-aware optimization
│   ├── sample_multi_target.py  # Multi-target generation
│   ├── run_multi_target.sh     # Run generation
│   ├── run_cma.sh              # CMA-ES optimization
│   ├── run_experiment.py       # Comprehensive experiments
│   ├── summarize.py            # Summarize docking scores
│   ├── unidock_client.py       # Uni-Dock client
│   └── gnina_wrapper.py        # GNINA wrapper
├── data/                       # Data files
│   ├── dict.txt                # Atom vocabulary
│   └── chembl_ring_systems.pkl # Ring system reference
├── extract_latent.py           # Latent space extraction
├── train_pretrain.sh           # Pretraining script
└── train_pretrain_flow.sh      # Pretraining with flow
```

---

## 🛠️ Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8
- RDKit >= 2022.03

### Setup Environment

```bash
# Create conda environment
conda create -n themol python=3.9
conda activate themol

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install rdkit-pypi numpy pandas scipy tqdm cmaes pyzmq

# Install unicore
pip install unicore

# Clone repository
git clone https://github.com/themolsubmission/TheMol.git
cd TheMol
```

### Optional Dependencies

```bash
# For xTB geometry optimization
conda install -c conda-forge xtb

# For PoseBusters evaluation
pip install posebusters

# For docking (Uni-Dock)
# Follow instructions at https://github.com/dptech-corp/Uni-Dock
```

---

## 📦 Data & Checkpoints

### Download Links

| Resource | Description | Link |
|----------|-------------|------|
| **Pretrained Model** | TheMol checkpoint | [Google Drive](https://drive.google.com/drive/folders/YOUR_MODEL_LINK) |
| **GEOM Dataset** | Pretraining data (LMDB) | [Google Drive](https://drive.google.com/drive/folders/YOUR_GEOM_LINK) |
| **MoleculeNet** | Property prediction data | [Google Drive](https://drive.google.com/drive/folders/YOUR_MOLNET_LINK) |
| **ADMET** | TDC ADMET benchmark data | [Google Drive](https://drive.google.com/drive/folders/YOUR_ADMET_LINK) |
| **CrossDocked** | Docking benchmark data | [Google Drive](https://drive.google.com/drive/folders/YOUR_DOCKING_LINK) |

### Data Preparation

```bash
# Download and extract data
cd TheMol

# Create data directories
mkdir -p datasets/geom
mkdir -p datasets/moleculenet
mkdir -p datasets/admet
mkdir -p checkpoints

# Download from Google Drive and place files:
# - datasets/geom/train.lmdb, valid.lmdb, test.lmdb
# - datasets/moleculenet/{bace,clintox,hiv,...}
# - datasets/admet/{ames,bbb,bioav,...}
# - checkpoints/checkpoint_last.pt
```

---

## 🚀 Quick Start

### 1. Pretraining

Train TheMol from scratch on GEOM dataset:

```bash
# Basic pretraining
bash train_pretrain.sh

# Pretraining with flow matching
bash train_pretrain_flow.sh
```

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `batch_size` | 256 | Batch size per GPU |
| `max_steps` | 7,000,000 | Total training steps |
| `warmup_steps` | 10,000 | Warmup steps |
| `n_gpu` | 2 | Number of GPUs |

---

## 🧪 Tasks

### Task 1: Unconditional Molecule Generation

Generate 3D molecules and evaluate with comprehensive metrics.

| Resource | Link |
|----------|------|
| Pretrained Checkpoint | [Google Drive](https://drive.google.com/drive/folders/YOUR_GENERATION_CKPT_LINK) |
| GEOM Dataset | [Google Drive](https://drive.google.com/drive/folders/YOUR_GEOM_LINK) |

```bash
cd generation

# Train generation model on GEOM
bash train_geom.sh

# Train on MIDI dataset
bash train_midi.sh

# Train with flow matching
bash train_midi_flow.sh
```

**Evaluate generated molecules:**

```bash
python evaluate.py \
    --checkpoint_path /path/to/checkpoint.pt \
    --num_samples 10000 \
    --output_dir ./results
```

**Evaluation Metrics:**
- **Validity**: RDKit sanitization success rate
- **Stability**: Bond-based and valence-based stability
- **REOS**: Structural alerts (Glaxo, Dundee rules)
- **Ring Systems**: Out-of-distribution ring ratio
- **Energy**: MMFF energy JS divergence
- **PoseBusters**: 3D geometry quality
- **Geometry**: xTB optimization metrics

---

### Task 2: Property Prediction (MoleculeNet)

Fine-tune TheMol on MoleculeNet benchmarks.

| Resource | Link |
|----------|------|
| Pretrained Checkpoint | [Google Drive](https://drive.google.com/drive/folders/YOUR_MOLNET_CKPT_LINK) |
| MoleculeNet Dataset | [Google Drive](https://drive.google.com/drive/folders/YOUR_MOLNET_DATA_LINK) |

```bash
cd property_pred

# Run grid search over hyperparameters
bash train.sh

# Summarize results
bash summarize.sh
```

**Supported Datasets:**

| Dataset | Task Type | Metric |
|---------|-----------|--------|
| BACE | Classification | AUC-ROC |
| BBBP | Classification | AUC-ROC |
| ClinTox | Multi-task Classification | AUC-ROC |
| HIV | Classification | AUC-ROC |
| Tox21 | Multi-task Classification | AUC-ROC |
| SIDER | Multi-task Classification | AUC-ROC |
| MUV | Multi-task Classification | AUC-PRC |
| ESOL | Regression | RMSE |
| FreeSolv | Regression | RMSE |
| Lipophilicity | Regression | RMSE |

**Hyperparameter Grid:**

```bash
learning_rates=(5e-5 8e-5 1e-4)
batch_sizes=(32 64)
pooler_dropouts=(0.0 0.1 0.2 0.5)
warmup_ratios=(0.0 0.06 0.1)
```

---

### Task 3: ADMET Prediction

Fine-tune TheMol on TDC ADMET benchmarks.

| Resource | Link |
|----------|------|
| Pretrained Checkpoint | [Google Drive](https://drive.google.com/drive/folders/YOUR_ADMET_CKPT_LINK) |
| ADMET Dataset | [Google Drive](https://drive.google.com/drive/folders/YOUR_ADMET_DATA_LINK) |

```bash
cd admet_pred

# Run grid search over hyperparameters
bash train.sh

# Summarize results with SOTA comparison
bash summarize.sh
```

**Supported ADMET Datasets:**

| Category | Datasets |
|----------|----------|
| Absorption | Caco-2, Bioavailability, PAMPA |
| Distribution | BBB, PPBR, VDss |
| Metabolism | CYP2C9/2D6/3A4 Inhibition/Substrate |
| Excretion | Clearance, Half-life |
| Toxicity | hERG, AMES, LD50, DILI |

---

### Task 4: Target-aware Molecule Optimization

Generate molecules optimized for specific protein targets using CMA-ES.

| Resource | Link |
|----------|------|
| Pretrained Checkpoint | [Google Drive](https://drive.google.com/drive/folders/YOUR_OPTIMIZATION_CKPT_LINK) |
| CrossDocked Dataset | [Google Drive](https://drive.google.com/drive/folders/YOUR_DOCKING_DATA_LINK) |

```bash
cd optimization

# Start Uni-Dock server (in a separate terminal)
python unidock_zmq_server.py --port 5555

# Run multi-target generation
bash run_multi_target.sh

# Or run with CMA-ES optimization
bash run_cma.sh

# Run comprehensive experiment
python run_experiment.py \
    --target_name CDK2 \
    --num_molecules 100 \
    --optimize_docking \
    --optimize_sa
```

**Optimization Objectives:**
- **Docking Score**: Uni-Dock or GNINA binding affinity
- **SA Score**: Synthetic accessibility (1-10, lower is better)
- **QED**: Quantitative estimate of drug-likeness
- **Validity**: Maintain chemical validity

**Summarize Results:**

```bash
python summarize.py --results_dir ./results
```

---

### Task 5: Latent Space Extraction

Extract molecular latent representations from pretrained TheMol model.

| Resource | Link |
|----------|------|
| Pretrained Checkpoint | [Google Drive](https://drive.google.com/drive/folders/YOUR_MODEL_LINK) |

```bash
# Extract latent from SDF file
python extract_latent.py \
    --sdf_path /path/to/molecules.sdf \
    --checkpoint_path /path/to/checkpoint.pt \
    --dict_path ./data/dict.txt \
    --output_path latent.pkl

# Extract latent from SMILES
python extract_latent.py \
    --smiles "CCO" \
    --checkpoint_path /path/to/checkpoint.pt \
    --dict_path ./data/dict.txt \
    --output_path latent.pkl

# Extract latent from directory of SDF files
python extract_latent.py \
    --sdf_dir /path/to/sdf_directory \
    --checkpoint_path /path/to/checkpoint.pt \
    --dict_path ./data/dict.txt \
    --output_path latents.pkl

# Extract with different aggregation methods
python extract_latent.py \
    --sdf_path /path/to/molecules.sdf \
    --aggregate mean \
    --output_path latent.pkl
```

**Aggregation Methods:**
| Method | Description |
|--------|-------------|
| `mean` | Mean pooling over atom latents (default) |
| `sum` | Sum pooling over atom latents |
| `cls` | Use first token (BOS) latent |
| `all` | Return all atom-level latents |

**Output Format:**
```python
# Output pickle contains:
{
    'smiles': str,                    # SMILES string
    'latent_mean': np.array,          # Aggregated latent mean [8]
    'latent_z': np.array,             # Sampled latent vector [8]
    'latent_mean_full': np.array,     # Per-atom latent means [N, 8]
    'latent_std_full': np.array,      # Per-atom latent stds [N, 8]
    'num_atoms': int,                 # Number of atoms
}
```

---

## 🔧 Configuration

### Model Architecture

```python
# Encoder configuration
encoder_embed_dim = 128
encoder_attention_heads = 8
encoder_ffn_embed_dim = 128
encoder_layers = 10

# Decoder configuration
decoder_layers = 5
decoder_ffn_embed_dim = 128
decoder_attention_heads = 8
```

### Training Configuration

```python
# Optimizer
optimizer = "adam"
adam_betas = (0.9, 0.99)
adam_eps = 1e-6
weight_decay = 1e-4

# Learning rate schedule
lr_scheduler = "polynomial_decay"
warmup_updates = 10000

# Loss weights
masked_token_loss = 1.0
masked_coord_loss = 10.0
masked_dist_loss = 10.0
```

---

## 🙏 Acknowledgements

We thank the authors of the following projects for their excellent work:

- [Uni-Mol](https://github.com/deepmodeling/Uni-Mol)
- [FlowMol](https://github.com/Dunni3/FlowMol)
- [GeoLDM](https://github.com/MinkaiXu/GeoLDM)
- [MiDi](https://github.com/cvignac/MiDi)

---

<p align="center">
  <i>If you have any questions, please open an issue or contact us.</i>
</p>

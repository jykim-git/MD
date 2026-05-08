# Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor

This repository provides the official implementation of the ICML 2026 paper:

**Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor**  
Jiyeon Kim, Byungju Lee, and Won-Yong Shin  
*Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)*

---

## Data Preparation

We provide three preprocessed datasets:

- **Dataset 1:** `liflow_...`
- **Dataset 2:** `amorphous_T_...`
- **Dataset 3:** `obelix_...`

In the dataset file names:

- `pos` denotes atomic positions.
- `temp` denotes temperature.
- `s_idx` denotes the target ion index.
- `y` denotes the target ionic transport property.
- `p` denotes atomic trajectory embeddings.

The preprocessed datasets can be downloaded from:

https://drive.google.com/drive/folders/1UlRrfTQqrjpfYO40GFvfztyoho5jn232?usp=sharing

After downloading the datasets, unzip the files and place them in the `data/` directory.

The raw datasets are available from the following sources:

- **Dataset 1:** https://zenodo.org/records/14889658
- **Dataset 2:** https://figshare.com/s/30601968f9244d8dffaa
- **Dataset 3:** https://github.com/NRC-Mila/OBELiX

To extract trajectory embeddings from atomic trajectories in the raw datasets, use:

```bash
notebooks/moment.ipynb
```

For example, this notebook can be used to generate files such as:

```bash
liflow_p_train_600K.pkl
```

To extract structure embeddings from atomic positions in the raw datasets, use:

```bash
notebooks/sevennet.ipynb
```

For example, this notebook can be used to generate files such as:

```bash
liflow_edge_attr_train.pkl
liflow_edge_index_train.pkl
liflow_node_attr_train.pkl
```

Both `notebooks/moment.ipynb` and `notebooks/sevennet.ipynb` assume that Dataset 1 is located in the `liflow_data/` directory, as described below.

---

## Directory Structure

```bash
liflow_data/  # Only necessary if you want to preprocess the raw data yourself
└── data/
    └── universal/

MD/  # This repository
├── data/
├── model/
├── notebooks/
├── preprocess/
├── results/
├── train/
└── utils/
```

---

## Installation

### Installation Tips for Using Preprocessed Data

1. Create a main conda environment.

2. Install PyTorch and related packages.  
   Please refer to the official PyTorch previous versions page:

   https://pytorch.org/get-started/previous-versions/

   For example:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

3. Install PyTorch Geometric and its dependencies.  
   Make sure that the PyTorch and CUDA versions are compatible.

```bash
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

pip install torch_cluster torch_scatter torch_sparse torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

4. Use NumPy 1.x:

```bash
pip install numpy==1.26.4
```

### Main Dependencies for Using Preprocessed Data

- Python 3.11.14
- NumPy 1.26.4
- PyTorch 2.1.2
- Torchvision 0.16.2
- Torchaudio 2.1.2
- torch-geometric 2.7.0
- torch_cluster 1.6.3
- torch_scatter 2.1.2
- torch_sparse 0.6.18
- torch_spline_conv 1.2.2

---

## Additional Installation for Extracting Structure Embeddings from Raw Data

To extract structure embeddings from raw data, additional packages and modifications are required.

1. In the main conda environment, install `sevenn` by following the official documentation:

   https://sevennet.readthedocs.io/en/latest/

   The `ase` package will be automatically installed with `sevenn`.

2. Downgrade `matscipy` to a compatible version, for example:

```bash
pip install matscipy==1.1.0
```

3. Locate `calculator.py` in the installed `sevenn` package.

   For example:

```bash
~/anaconda3/envs/<main_environment_name>/lib/python3.11/site-packages/sevenn/calculator.py
```

Then copy and paste the code in `utils/calculate2.py` from this repository to `SevenNetCalculator` in `calculator.py`.

4. Locate `sequential.py` in the installed `sevenn` package.

   For example:

```bash
~/anaconda3/envs/<main_environment_name>/lib/python3.11/site-packages/sevenn/nn/sequential.py
```

Modify the `forward` function in `AtomGraphSequential` so that the forward pass stops after the 11th layer:

```python
for i, module in enumerate(self):
    data = module(data)
    if i > 10:
        break
```

### Additional Dependencies for Extracting Structure Embeddings

- sevenn 0.12.0
- matscipy 1.1.0
- ase 3.27.0

---

## Additional Installation for Extracting Trajectory Embeddings from Raw Data

To extract trajectory embeddings from raw data, follow the instructions for MOMENT:

https://huggingface.co/AutonLab/MOMENT-1-small

For example:

1. Create a new conda environment.

2. Install `momentfm`.

3. Install `pandas` version 2.3.3 or earlier:

```bash
pip install "pandas<=2.3.3"
```

### Additional Dependencies for Extracting Trajectory Embeddings

- Python 3.11.14
- momentfm 0.1.4
- pandas 2.3.3

---

## Usage

### Model-Level Auxiliary Modality Learning

For model-level auxiliary modality learning on Dataset 1, refer to:

```bash
notebooks/model_level.ipynb
```

### Data-Level Auxiliary Modality Learning on Dataset 2

For data-level auxiliary modality learning on Dataset 2, refer to:

```bash
notebooks/data_level_amorphous.ipynb
```

### Data-Level Auxiliary Modality Learning on Dataset 3

For data-level auxiliary modality learning on Dataset 3, refer to:

```bash
notebooks/data_level_obelix.ipynb
```

Both `data_level_amorphous.ipynb` and `data_level_obelix.ipynb` require the models saved after running `model_level.ipynb`.

The resulting log files will be saved in the `results/` directory, and the resulting models will be saved in the `model/` directory.
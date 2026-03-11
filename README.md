# Sentinel-2 Forest Browning Monitoring

Code for **Country-wide, high-resolution monitoring of forest browning with Sentinel-2**

Authors: Samantha Biegel, David Brüggemann, Francesco Grossi, Michele Volpi, Konrad Schindler, Benjamin Stocker


## Repository contents

This repository contains:
- the `forest_browning` Python package (`src/forest_browning`),
    - `src/forest_browning/train.py`: model training
    - `src/forest_browning/inference.py`: inference/anomaly outputs
    - `src/forest_browning/dataset.py`: dataset utilities
    - `src/forest_browning/mlp.py`: model architecture
    - `src/forest_browning/config.py`: constants
    -
- analysis notebook (`notebooks/`),
    - `notebooks/plot_results.ipynb`: generates figures in `figs/`
- event polygons and summary CSVs (`data/`),
- model checkpoints (`checkpoints/`).

---

## Installation

```sh
git clone git@github.com:SamanthaBiegel/s2-forest-browning-monitoring.git
cd s2-forest-browning-monitoring

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install -r requirements.txt -e . --group dev

pre-commit install
pre-commit run --all-files
```


---

## Dataset creation (external repo: [`swiss-ndvi-processing`](https://github.com/geco-bern/swiss-ndvi-processing))

The processed dataset used by this project is created in a different repository: `swiss-ndvi-processing`, using the scripts in its `processing/` folder.

### 1) Clone and set up `swiss-ndvi-processing`

```sh
git clone git@github.com:geco-bern/swiss-ndvi-processing.git
cd swiss-ndvi-processing
```

Use the provided Conda environment:

```sh
conda env create -f environment.yml
conda activate <env-name-from-environment.yml>
```

Then install the package in editable mode:

```sh
pip install -e .
```

### 2) Configure processing paths

Edit:

- `swiss-ndvi-processing/processing/config.py`

to set output paths for your machine.

### 3) Run processing pipeline in order

From `swiss-ndvi-processing` root:

```sh
python processing/1_extract_swisstopo_dataset.py
python processing/2_transpose_swisstopo_dataset.py
python processing/3_add_dates.py
python processing/4_load_dem_2m.py
python processing/5_dem_2m_zarr_to_geotiff.py
python processing/6_create_dem_features.py
python processing/7_add_vegetation_height.py
python processing/8_add_tree_species.py
python processing/9_add_habitat.py
python processing/10_add_missingness.py
python processing/11_add_forest_mix_rate.py
python processing/12_merge_features.py
```

### 4) Output dataset

The final output dataset is a Zarr dataset with two versions:
- `ndvi_dataset_temporal.zarr`: includes the full NDVI time series with chunking along the temporal dimension (shape: `(num_forest_pixels, num_timesteps)`) as well as the feature arrays
- `ndvi_dataset_spatial.zarr`: includes the full NDVI time series with chunking along the spatial dimension (shape: `(num_timesteps, n_forest_pixels)`)

A forest mask is also generated as a NumPy array (`forest_mask.npy`), which is used to map between the 1D dataset and the original 2D spatial layout.

Adjust the paths in `src/forest_browning/config.py` to point to your local output locations for these datasets.

---

## Running this project

From `s2-forest-browning-monitoring`:
```sh
source .venv/bin/activate
```

First pre-shuffle the training dataset:

```sh
python -m forest_browning.shuffle_train_data --input_zarr /path/to/ndvi_dataset_temporal.zarr --output_zarr /path/to/ndvi_dataset_filtered_shuffled.zarr
```

Then run training to generate vegetation cycle parameters:

```sh
python -m forest_browning.train --data_path /path/to/ndvi_dataset_filtered_shuffled.zarr --output_dir /path/to/checkpoints
```

Finally, run inference to generate anomaly scores:

```sh
python -m forest_browning.inference --model_checkpoint checkpoints/encoder.pt --data_path /path/to/ndvi_dataset_temporal.zarr --output_dir /path/to/ndvi_dataset_temporal.zarr
```

To obtain anomaly scores in the spatial format for fast per-day retrieval, run:

```sh
python -m forest_browning.rechunk_output.py --source_zarr /path/to/ndvi_dataset_temporal.zarr --target_zarr /path/to/ndvi_dataset_spatial.zarr
```

For figure reproduction, run:

- `notebooks/plot_results.ipynb`

Figures are saved to `figs/`.

---

## Project structure

```text
s2-forest-browning-monitoring/
├── src/
│   └── forest_browning/
├── notebooks/
├── data/
│   └── event_polygons/
├── checkpoints/
├── figs/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

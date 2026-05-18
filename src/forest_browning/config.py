"""Set constants."""

import os
from pathlib import Path

from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine


# Paths
DATA_DIR = os.getenv("FOREST_BROWNING_DATA_DIR")
if not DATA_DIR:
    raise RuntimeError(
        "Missing FOREST_BROWNING_DATA_DIR. Set it to your local data directory, "
        "for example: export FOREST_BROWNING_DATA_DIR=/path/to/data"
    )
DATA_DIR = os.path.abspath(os.path.expanduser(DATA_DIR))
TEMPORAL_DATASET_ZARR = f"{DATA_DIR}/ndvi_dataset_temporal.zarr"
SPATIAL_DATASET_ZARR = f"{DATA_DIR}/ndvi_dataset_spatial.zarr"
FOREST_MASK = f"{DATA_DIR}/forest_mask.npy"
DASK_LOCAL_DIRECTORY = f"{DATA_DIR}/dask_worker_space"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_DATA_DIR = PROJECT_ROOT / "data"

TREE_SPECIES_PATH = f"{DATA_DIR}/tree_species_map_aoa_raster.tif"

# External services
SERVICE_URL = "https://data.geo.admin.ch/api/stac/v0.9/"

# Reference grid
PX = 10.0
REF_BBOX = BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0)
REF_BBOX_4326 = BoundingBox(left=5.70, bottom=45.8, right=10.6, top=47.95)
REF_WIDTH = int((REF_BBOX.right - REF_BBOX.left) / PX)
REF_HEIGHT = int((REF_BBOX.top - REF_BBOX.bottom) / PX)
REF_TRANSFORM = Affine(PX, 0.0, REF_BBOX.left, 0.0, -PX, REF_BBOX.top)
REF_CRS = CRS.from_epsg(2056)

# Data loading
CHUNK_SIZE = 4000

# NDVI / NDSI processing
INVALID = -(2**15)  # Filtered out pixels, e.g. cloud shadows
NO_COVERAGE = 2**15 - 1  # Pixels with no data for the given time step
NDVI_SCALE = 10000.0
NDVI_MAX = 1.0
NDVI_MIN = -0.1
NDSI_SNOW_MIN = 4300
NDSI_SNOW_MAX = 10000

# Species / habitat encoding
N_TREE_SPECIES = 17
N_HABITATS = 46
INVALID_SPECIES_CODE = 255
MISSING_SPECIES_CODE = 16

# Model architecture
MODEL_D_OUT = 18
MODEL_N_BLOCKS = 8
MODEL_D_BLOCK = 256
MODEL_SPECIES_EMB_DIM = 4
MODEL_HABITAT_EMB_DIM = 8

# Anomaly detection
ANOMALY_FILL_VALUE = 127
ANOMALY_MASKED_VALUE = -128
IQR_MULTIPLIER = 1.5

"""Set constants."""

from rasterio.coords import BoundingBox
from rasterio.transform import Affine


TEMPORAL_DATASET_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
SPATIAL_DATASET_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_spatial.zarr"
FOREST_MASK = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

REF_BBOX = BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0)
REF_BBOX_4326 = BoundingBox(left=5.70, bottom=45.8, right=10.6, top=47.95)

PX = 10.0

REF_WIDTH = int((REF_BBOX.right - REF_BBOX.left) / PX)
REF_HEIGHT = int((REF_BBOX.top - REF_BBOX.bottom) / PX)

REF_TRANSFORM = Affine(PX, 0.0, REF_BBOX.left, 0.0, -PX, REF_BBOX.top)

CHUNK_SIZE = 4000

SERVICE_URL = "https://data.geo.admin.ch/api/stac/v0.9/"

INVALID = -(2**15)  # Filtered out pixels, e.g. cloud shadows
NO_COVERAGE = 2**15 - 1  # Pixels with no data for the given time step

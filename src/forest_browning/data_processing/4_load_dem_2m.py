"""Load the SwissALTI3D DEM data at 2m resolution and store in a Zarr array."""

import os

import rasterio
import zarr
from rasterio.coords import BoundingBox
from rasterio.transform import Affine
from tqdm import tqdm

from forest_browning.config import DATA_DIR, REF_BBOX, REPO_DATA_DIR


# Reference raster metadata with 2m resolution
ref_height_2m = int((REF_BBOX.top - REF_BBOX.bottom) / 2.0)
ref_width_2m = int((REF_BBOX.right - REF_BBOX.left) / 2.0)
ref_transform_2m = Affine(2.0, 0.0, REF_BBOX.left, 0.0, -2.0, REF_BBOX.top)

os.makedirs(f"{DATA_DIR}/tmp", exist_ok=True)

# Create a Zarr array to store the DEM data
compressors = zarr.codecs.BloscCodec(
    cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle
)
zarr_array = zarr.create_array(
    name="dem_2m",
    store=f"{DATA_DIR}/tmp/full_dem_2m.zarr",
    shape=(ref_height_2m, ref_width_2m),
    chunks=(500, 500),
    dtype="float32",
    compressors=compressors,
    zarr_format=3,
)


def bounds_intersect(a: BoundingBox, b: BoundingBox) -> bool:
    """Check if two bounding boxes intersect."""
    return not (
        a.right <= b.left or a.left >= b.right or a.top <= b.bottom or a.bottom >= b.top
    )


zarr_array.attrs["transform"] = ref_transform_2m.to_gdal()
zarr_array.attrs["crs"] = "EPSG:2056"

# Read the URLs from the CSV file
with open(REPO_DATA_DIR / "ch.swisstopo.swissalti3d-15IhqQCR.csv") as f:
    urls = [line.strip() for line in f if line.strip()]

for url in tqdm(urls, desc="Processing DEM tiles"):
    with rasterio.open(url) as src:
        if not bounds_intersect(src.bounds, REF_BBOX):
            continue

        data = src.read(1)

        # Compute the offset in the reference raster grid
        row_off = int((ref_transform_2m.f - src.bounds.top) / 2.0)
        col_off = int((src.bounds.left - ref_transform_2m.c) / 2.0)

        # Compute the destination slices in the Zarr array
        dest_row = slice(row_off, row_off + data.shape[0])
        dest_col = slice(col_off, col_off + data.shape[1])

        # Ensure the destination slices are within bounds
        if (
            dest_row.start < 0
            or dest_col.start < 0
            or dest_row.stop > ref_height_2m
            or dest_col.stop > ref_width_2m
        ):
            continue

        zarr_array[dest_row, dest_col] = data

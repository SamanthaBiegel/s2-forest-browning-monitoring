#!/usr/bin/env python
"""Run the full dataset creation pipeline."""

import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Extract Swisstopo dataset", "1_extract_swisstopo_dataset"),
    ("Transpose Swisstopo dataset", "2_transpose_swisstopo_dataset"),
    ("Add dates", "3_add_dates"),
    ("Load DEM 2m", "4_load_dem_2m"),
    ("DEM 2m Zarr to GeoTIFF", "5_dem_2m_zarr_to_geotiff"),
    ("Create DEM features", "6_create_dem_features"),
    ("Add vegetation height", "7_add_vegetation_height"),
    ("Add tree species", "8_add_tree_species"),
    ("Add habitat", "9_add_habitat"),
    ("Add missingness", "10_add_missingness"),
    ("Add forest mix rate", "11_add_forest_mix_rate"),
    ("Merge features", "12_merge_features"),
]


def main():
    """Execute all dataset creation steps in order."""
    total = len(STEPS)
    script_dir = Path(__file__).resolve().parent

    for i, (description, script_name) in enumerate(STEPS, 1):
        try:
            print(f"[{i}/{total}] {description}...", end=" ", flush=True)
            script_path = script_dir / f"{script_name}.py"
            _ = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            print("  Success")
        except subprocess.CalledProcessError as e:
            print(f"  Error: {e.stderr}")
            return 1
        except Exception as e:
            print(f"  Error: {e}")
            return 1

    print(f"\n Dataset creation complete ({total}/{total} steps successful).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

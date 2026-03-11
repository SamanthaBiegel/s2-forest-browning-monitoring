"""A script to filter and shuffle NDVI/NDSI Zarr datasets for training."""

import argparse

import numpy as np
import zarr
from torch.utils.data import DataLoader
from tqdm import tqdm


class ShuffleDataset:
    """A dataset for shuffling and filtering NDVI/NDSI Zarr datasets.

    Attributes:
        file_path (str): Path to the input Zarr store.
        feat_array (zarr.Array): Zarr array containing the merged features.
        ndvi (zarr.Array): Zarr array containing NDVI values.
        ndsi (zarr.Array): Zarr array containing NDSI values.
        dataset_len (int): Total number of samples in the dataset.
        timesteps (int): Number of time steps in the NDVI/NDSI arrays.
        n_features (int): Number of features in the merged features array.
    """

    def __init__(self, file_path):
        """Initializes the ShuffleDataset by loading the NDVI, NDSI, and merged features arrays from the specified Zarr store.

        Args:
            file_path (str): Path to the input Zarr store.
        """
        self.file_path = file_path
        zarr_store = zarr.open(file_path, mode="r", zarr_format=3)
        self.feat_array = zarr_store["merged_features"]

        self.ndvi = zarr_store["ndvi"]
        self.ndsi = zarr_store["ndsi"]
        self.dataset_len = self.ndvi.shape[0]
        self.timesteps = self.ndvi.shape[1]
        self.n_features = self.feat_array.shape[1]

    def __getitem__(self, idx):
        """Returns the NDVI, NDSI, and merged features for the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the NDVI, NDSI, and merged features for the specified index.
        """
        ndvi = self.ndvi[idx]
        ndsi = self.ndsi[idx]
        features = self.feat_array[idx]
        return ndvi, ndsi, features

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return self.dataset_len


def write_shuffled_copy(
    dataloader, target_zarr, n_samples, n_timesteps, n_features, chunk_rows=8192
):
    """A function to write a shuffled and filtered copy of the NDVI/NDSI datasets to a new Zarr store.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the ShuffleDataset.
        target_zarr (str): Path to the output Zarr store.
        n_samples (int): Number of samples in the output dataset.
        n_timesteps (int): Number of time steps in the NDVI/NDSI arrays.
        n_features (int): Number of features in the merged features array.
        chunk_rows (int, optional): Number of rows per chunk in the output Zarr arrays. Defaults to 8192.
    """
    root = zarr.open_group(target_zarr, mode="w", zarr_format=3)
    ndvi_out = root.create_array(
        "ndvi",
        shape=(n_samples, n_timesteps),
        chunks=(chunk_rows, n_timesteps),
        dtype=np.int16,
        compressors=[],
    )
    feat_out = root.create_array(
        "merged_features",
        shape=(n_samples, n_features),
        chunks=(chunk_rows, n_features),
        dtype=dataloader.dataset.feat_array.dtype,
        compressors=[],
    )

    offset = 0
    for batch in tqdm(dataloader, desc="Writing shuffled NDVI", total=len(dataloader)):
        ndvi, ndsi, feat = batch
        ndvi = ndvi.numpy().astype(np.int16)
        ndsi = ndsi.numpy().astype(np.int16)
        feat = feat.numpy()
        mask = (ndsi > 4300) & (ndsi < 10000)
        ndvi[mask] = -(2**15)
        ndvi_out[offset : offset + ndvi.shape[0], :] = ndvi
        feat_out[offset : offset + feat.shape[0], :] = feat
        offset += ndvi.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter and shuffle NDVI/NDSI Zarr datasets."
    )

    parser.add_argument(
        "--input_zarr",
        type=str,
        required=True,
        help="Path to the input Zarr store (e.g., /path/to/ndvi_dataset_temporal.zarr)",
    )
    parser.add_argument(
        "--output_zarr",
        type=str,
        required=True,
        help="Path to the output Zarr store (e.g., /path/to/ndvi_dataset_filtered_shuffled.zarr)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for the DataLoader (default: 512)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of workers for the DataLoader (default: 32)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Prefetch factor for the DataLoader (default: 4)",
    )

    args = parser.parse_args()

    # Load dataset
    ds = ShuffleDataset(args.input_zarr)

    # Initialize dataloader
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
    )

    # Process and write
    write_shuffled_copy(
        loader,
        args.output_zarr,
        n_samples=len(ds),
        n_timesteps=ds.timesteps,
        n_features=ds.n_features,
        chunk_rows=args.batch_size,
    )

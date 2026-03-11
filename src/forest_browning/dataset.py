"""A dataset for loading Zarr data, with support for chunked streaming and shuffling."""

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import IterableDataset


# Rescale input (stats obtained from full dataset)
MEANS = {
    "dem": 1071.1402587890625,
    "slope": 26.603004455566406,
    "easting": 0.01555223111063242,
    "northing": -0.06615273654460907,
    "twi": 2.7378883361816406,
    "mean_curv": 0.00045366896665655077,
    "profile_curv": 0.00016974707250483334,
    "plan_curv": 0.002614102093502879,
    "tri": 2.885868787765503,
    "roughness": 2.9208521842956543,
    "median_forest_height": 24.493475,
    "forest_mix_rate": 0.2507593148494662,
}
STDS = {
    "dem": 450.6628112792969,
    "slope": 14.156021118164062,
    "easting": 0.6557915806770325,
    "northing": 0.6876032948493958,
    "twi": 1.7116827964782715,
    "mean_curv": 0.005201074760407209,
    "profile_curv": 0.005349245388060808,
    "plan_curv": 0.0478176586329937,
    "tri": 2.302125930786133,
    "roughness": 2.2183449268341064,
    "median_forest_height": 6.8817625,
    "forest_mix_rate": 0.6688715032184841,
}


class ZarrDataset(IterableDataset):
    """An IterableDataset that reads from a Zarr file, shuffles the order of the samples, and yields batches of data."""

    all_features = [
        "dem",
        "slope",
        "easting",
        "northing",
        "twi",
        "tri",
        "mean_curv",
        "profile_curv",
        "plan_curv",
        "roughness",
        "median_forest_height",
        "forest_mix_rate",
        "tree_species",
        "habitat",
    ]

    def __init__(
        self,
        file_path,
        features=None,
        chunked=True,
        include_ndsi=False,
        batch_size=512,
        chunk_size=8192,
        shuffle_chunks=True,
        seed=42,
    ):
        """Initialize the dataset.

        Args:
            file_path (str): Path to the single Zarr file.
            features (list of str, optional): List of feature names to include. Defaults to None (all features).
            chunked (bool, optional): Whether to use chunked streaming mode. Defaults to True.
            include_ndsi (bool, optional): Whether to yield the NDSI array alongside NDVI and features.
            batch_size (int, optional): Size of each batch. Defaults to 512.
            chunk_size (int, optional): Size of each chunk. Defaults to 8192.
            shuffle_chunks (bool, optional): Whether to shuffle chunks. Defaults to True.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__()
        self.file_path = file_path
        self.features = features if features is not None else self.all_features
        self.chunked = chunked
        self.include_ndsi = include_ndsi
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        self.base_seed = seed

        store = zarr.open(self.file_path, mode="r", zarr_format=3)

        self.ndvi = store["ndvi"]
        self.feat = store["merged_features"]
        self.mapping_features = store["merged_features"].attrs["feature_columns"]
        self.missingness = store["missingness"][:]
        if self.include_ndsi:
            self.ndsi = store["ndsi"]

        # Load and process dates
        self.dates = np.array(
            [pd.to_datetime(d.decode("utf-8")) for d in store["dates"][:]]
        )
        dtindex = pd.DatetimeIndex(self.dates)
        self.doy = dtindex.dayofyear.to_numpy()
        is_leap = dtindex.is_leap_year.astype(int)
        # Normalize day of year to [0, 1]
        self.t = (self.doy - 1) / (365 + is_leap)

        self.dataset_len = self.ndvi.shape[0]
        self.timesteps = self.ndvi.shape[1]
        self.n_chunks = int(np.ceil(self.dataset_len / self.chunk_size))
        self.num_features = [
            f for f in self.features if f not in ["tree_species", "habitat"]
        ]

        # Total number of minibatches per epoch
        self.n_batches = int(
            (self.dataset_len + self.batch_size - 1) // self.batch_size
        )
        self.nr_num_features = len(self.num_features)
        self.num_feature_indices = [
            i for f in self.num_features for i in self.mapping_features[f]
        ]
        self.nr_tree_species = 17
        self.nr_habitats = 46

    def set_epoch(self, epoch):
        """Set the epoch for reproducible shuffling. Should be called at the beginning of each epoch in training.

        Args:
            epoch (int): The current epoch number.
        """
        self.epoch = epoch
        self.rng = np.random.default_rng(self.base_seed + epoch)

    def __iter__(self):
        """Yield batches of data.

        Yields:
            tuple: A tuple containing the NDVI, NDSI (if included), and features for each batch.
        """
        if not self.chunked:
            for i in range(self.dataset_len):
                yield self.__getitem__(i)
            return

        worker_info = torch.utils.data.get_worker_info()
        if hasattr(self, "epoch"):
            epoch = self.epoch
        else:
            epoch = 0

        if worker_info is not None:
            wid = worker_info.id
            nworkers = worker_info.num_workers
        else:
            wid = 0
            nworkers = 1

        # Create a unique seed for each worker and epoch
        seed = self.base_seed + wid + epoch * 1000
        self.rng = np.random.default_rng(seed)

        store = zarr.open(self.file_path, mode="r", zarr_format=3)
        self.ndvi = store["ndvi"]
        self.feat = store["merged_features"]
        if self.include_ndsi:
            self.ndsi = store["ndsi"]

        chunk_indices = np.arange(self.n_chunks)
        if self.shuffle_chunks:
            # Shuffle chunk indices at the start of each epoch
            self.rng.shuffle(chunk_indices)

        # Split chunk indices among workers
        chunk_indices = chunk_indices[wid::nworkers]

        for cid in chunk_indices:
            # Compute start and stop indices for the current chunk
            start = cid * self.chunk_size
            # Make sure to not go out of bounds on the last chunk
            stop = min((cid + 1) * self.chunk_size, self.dataset_len)

            ndvi_chunk = np.asarray(self.ndvi[start:stop])
            feat_chunk = np.asarray(self.feat[start:stop])
            if self.include_ndsi:
                ndsi_chunk = np.asarray(self.ndsi[start:stop])

            # Shuffle samples within the chunk
            order = self.rng.permutation(len(ndvi_chunk))
            ndvi_chunk = ndvi_chunk[order]
            feat_chunk = feat_chunk[order]
            if self.include_ndsi:
                ndsi_chunk = ndsi_chunk[order]

            for i in range(0, len(ndvi_chunk), self.batch_size):
                # Compute end index for the current batch
                j = min(i + self.batch_size, len(ndvi_chunk))

                # Yield the current batch as PyTorch tensors
                if self.include_ndsi:
                    yield (
                        torch.from_numpy(ndvi_chunk[i:j]).float(),
                        torch.from_numpy(ndsi_chunk[i:j]).float(),
                        torch.from_numpy(feat_chunk[i:j]).float(),
                    )
                else:
                    yield (
                        torch.from_numpy(ndvi_chunk[i:j]).float(),
                        torch.from_numpy(feat_chunk[i:j]).float(),
                    )

    def __getitem__(self, idx):
        """Get the NDVI, NDSI (if included), and features for a single sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the NDVI, NDSI (if included), and features for the specified index.
        """
        ndvi = torch.from_numpy(self.ndvi[idx])
        features = torch.from_numpy(self.feat[idx]).float()

        if self.include_ndsi:
            ndsi = torch.from_numpy(self.ndsi[idx])
            return ndvi, ndsi, features

        return ndvi, features

    def __len__(self):
        """Return total number of minibatches if chunked, or dataset len if not."""
        if self.chunked:
            return self.n_batches
        return self.dataset_len

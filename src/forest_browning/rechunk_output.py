"""Transpose NDVI anomaly data from (N, T) to (T, N) using Dask for efficient processing."""

import argparse

import dask
import dask.array as da
import zarr
from dask.distributed import Client, LocalCluster

# Set global Dask config for memory spilling
dask.config.set({"distributed.worker.memory.spill-compression": "lz4"})


def transpose_zarr(
    source_zarr: str,
    target_zarr: str,
    component: str = "ndvi",
    n_slices: int = 3,
    client: Client | None = None,
) -> None:
    """Transposes a Zarr array from (N, T) to (T, N) in chunks to manage memory.

    Args:
        source_zarr (str): Path to the input Zarr store.
        target_zarr (str): Path to the output Zarr store.
        component (str, optional): The specific array/group within the Zarr store to transpose. Defaults to "ndvi".
        n_slices (int, optional): Number of slices to split the N dimension into for processing. Defaults to 3.
        client (Client, optional): The Dask distributed client, used to restart workers between slices to clear memory. Defaults to None.
    """
    src = da.from_zarr(source_zarr, component=component)
    N, T = src.shape

    # Fixed: Using 'target_zarr' instead of the global 'TRANSPOSED_ZARR'
    empty = da.zeros((T, N), chunks=(1, N))
    empty.to_zarr(target_zarr, component=component, compute=False, overwrite=True)

    target_group = zarr.open(target_zarr, mode="a")
    target_array = target_group[component]

    step = N // n_slices
    for i in range(n_slices):
        start = i * step
        end = (i + 1) * step if i < n_slices - 1 else N

        src_slice = src[start:end, :]

        dst_slice = src_slice.T.rechunk(chunks=(1, end - start))

        dst_slice.to_zarr(
            target_array, region=(slice(0, T), slice(start, end)), compute=True
        )

        if client:
            client.restart()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transpose Zarr datasets using Dask.")

    # Required arguments for file paths
    parser.add_argument(
        "--source_zarr", type=str, required=True, help="Path to the source Zarr store."
    )
    parser.add_argument(
        "--target_zarr", type=str, required=True, help="Path to the target Zarr store."
    )

    # Optional arguments for execution configuration
    parser.add_argument(
        "--dask_dir",
        type=str,
        default="/tmp/dask_worker_space",
        help="Local directory for Dask worker space.",
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=["anomalies", "anomaly_scores"],
        help="List of components to transpose (e.g., anomalies anomaly_scores).",
    )
    parser.add_argument(
        "--n_slices",
        type=int,
        default=3,
        help="Number of slices to split the transposition into (default: 3).",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4, help="Number of Dask workers (default: 4)."
    )
    parser.add_argument(
        "--memory_limit",
        type=str,
        default="20GB",
        help="Memory limit per Dask worker (default: 20GB).",
    )

    args = parser.parse_args()

    # Initialize Dask Cluster and Client
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=1,
        processes=True,
        memory_limit=args.memory_limit,
        local_directory=args.dask_dir,
    )
    client = Client(cluster)

    # Process each component provided in the arguments
    for comp in args.components:
        print(f"Transposing component: {comp}")
        transpose_zarr(
            source_zarr=args.source_zarr,
            target_zarr=args.target_zarr,
            component=comp,
            n_slices=args.n_slices,
            client=client,
        )

    print("Transposition complete.")
    client.close()
    cluster.close()

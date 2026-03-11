"""Run inference using a trained encoder to predict seasonal NDVI parameters and anomalies for each pixel in the dataset."""

import argparse

import numpy as np
import torch
import zarr
from tqdm import tqdm

from forest_browning.config import CHUNK_SIZE, INVALID, NO_COVERAGE
from forest_browning.dataset import ZarrDataset
from forest_browning.mlp import MLPWithEmbeddings
from forest_browning.train import (
    MEANS,
    STDS,
    double_logistic_function,
)


def rectify_parameters(params):
    """Rectify the predicted parameters to ensure they are in a valid range and order for the double logistic function.

    Args:
        params (torch.Tensor): Tensor of shape (batch_size, 6) containing the predicted parameters [sos, mat_minus_sos, sen, eos_minus_sen, M, m].

    Returns:
        torch.Tensor: Rectified parameters.
    """
    inverted_mask = params[:, 0] > params[:, 2]

    rec_params = params.clone()
    rec_params[inverted_mask, 0] = params[inverted_mask, 2]
    rec_params[inverted_mask, 1] = params[inverted_mask, 3]
    rec_params[inverted_mask, 2] = params[inverted_mask, 0]
    rec_params[inverted_mask, 3] = params[inverted_mask, 1]
    rec_params[inverted_mask, 4] = (
        params[inverted_mask, 5] + params[inverted_mask, 5] - params[inverted_mask, 4]
    )

    return rec_params


def chunk_iterator(zarr_array, chunk_size):
    """Yield slices for iterating over a Zarr array in chunks of a specified size.

    Args:
        zarr_array (zarr.Array): The Zarr array to iterate over.
        chunk_size (int): The size of each chunk.

    Yields:
        slice: A slice object representing the current chunk.
    """
    n = zarr_array.shape[0]
    for i in range(0, n, chunk_size):
        yield slice(i, min(i + chunk_size, n))


def inference(args):
    """Run inference using a trained encoder to predict seasonal NDVI parameters and anomalies for each pixel in the dataset.

    Args:
        args: Command-line arguments containing paths to the encoder, dataset, and output location, as well as feature selection and device configuration.
    """
    ds = ZarrDataset(args.data_path)

    encoder = MLPWithEmbeddings(
        d_num=ds.nr_num_features,
        d_out=18,
        n_blocks=8,
        d_block=256,
        dropout=0.0,
        skip_connection=True,
        n_species=ds.nr_tree_species,
        species_emb_dim=4,
        n_habitats=ds.nr_habitats,
        habitat_emb_dim=8,
    ).to(args.device)
    encoder.load_state_dict(torch.load(args.encoder_path))
    encoder.eval()

    means_pt = (
        torch.tensor([MEANS[f] for f in ds.num_features]).to(args.device).unsqueeze(0)
    )
    stds_pt = (
        torch.tensor([STDS[f] for f in ds.num_features]).to(args.device).unsqueeze(0)
    )

    num_columns = [
        column
        for feat_name in ds.num_features
        for column in ds.mapping_features[feat_name]
    ]
    species_columns = ds.mapping_features["tree_species"]
    habitat_columns = ds.mapping_features["habitat"]

    N = ds.dataset_len
    T = ds.timesteps
    print(f"Number of samples: {N}")

    root_params = zarr.open_group(args.output_path, mode="a")
    feat_grp = root_params.create_group("params")

    feat_grp.create_array(
        name="params_lower",
        shape=(N, 6),
        chunks=(CHUNK_SIZE, 6),
        dtype="float32",
    )

    feat_grp.create_array(
        name="params_median",
        shape=(N, 6),
        chunks=(CHUNK_SIZE, 6),
        dtype="float32",
    )

    feat_grp.create_array(
        name="params_upper",
        shape=(N, 6),
        chunks=(CHUNK_SIZE, 6),
        dtype="float32",
    )

    description = (
        "Double logistic function parameters:\n"
        "- sos: start of season (day of year)\n"
        "- mat_minus_sos: duration of green-up\n"
        "- sen: start of senescence\n"
        "- eos_minus_sen: duration of senescence\n"
        "- M: maximum NDVI\n"
        "- m: minimum NDVI"
    )

    feat_grp["params_lower"].attrs["description"] = (
        "Lower bound parameters of the seasonal NDVI cycle.\n" + description
    )
    feat_grp["params_median"].attrs["description"] = (
        "Median parameters of the seasonal NDVI cycle.\n" + description
    )
    feat_grp["params_upper"].attrs["description"] = (
        "Upper bound parameters of the seasonal NDVI cycle.\n" + description
    )
    feat_grp.attrs["encoder_path"] = args.encoder_path

    feat_grp_preds = root_params.create_group("ndvi_preds")
    feat_grp_preds.create_array(
        name="ndvi_pred_lower",
        shape=(N, T),
        chunks=(CHUNK_SIZE, T),
        dtype="int16",
    )
    feat_grp_preds.create_array(
        name="ndvi_pred_median",
        shape=(N, T),
        chunks=(CHUNK_SIZE, T),
        dtype="int16",
    )
    feat_grp_preds.create_array(
        name="ndvi_pred_upper",
        shape=(N, T),
        chunks=(CHUNK_SIZE, T),
        dtype="int16",
    )

    root_params.create_array(
        name="anomalies",
        shape=(N, T),
        chunks=(CHUNK_SIZE, T),
        dtype="int8",
        fill_value=127,
        compressors=zarr.codecs.BloscCodec(
            cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.bitshuffle
        ),
    )
    root_params["anomalies"].attrs["description"] = (
        "NDVI anomaly values (int8) for each pixel in the reference grid. 0 means no anomaly, "
        "1 means positive anomaly, -1 means negative anomaly. Missing values (no coverage, no forest) are indicated by 127."
        "Masked values (clouds, shadows, snow, outliers) are indicated by -128."
    )
    root_params["anomalies"].attrs["encoder_path"] = args.encoder_path

    root_params.create_array(
        name="anomaly_scores",
        shape=(N, T),
        chunks=(CHUNK_SIZE, T),
        dtype="float32",
        fill_value=np.nan,
        compressors=zarr.codecs.BloscCodec(
            cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.bitshuffle
        ),
    )
    root_params["anomaly_scores"].attrs["encoder_path"] = args.encoder_path

    t = torch.tensor(ds.t).to(args.device).float()

    with torch.inference_mode():
        for i, slc in enumerate(
            tqdm(
                chunk_iterator(ds.feat_array, CHUNK_SIZE),
                total=(N + CHUNK_SIZE - 1) // CHUNK_SIZE,
            )
        ):
            feat = torch.from_numpy(ds.feat_array[slc, :]).to(args.device).float()
            ndvi = torch.from_numpy(ds.ndvi[slc, :]).to(args.device).float()
            ndsi = torch.from_numpy(ds.ndsi[slc, :]).to(args.device).float()

            feat_num = feat[:, num_columns]
            feat_species = feat[:, species_columns].int()
            feat_habitat = feat[:, habitat_columns].int()
            feat_species[feat_species == 255] = 16

            # Standardize input
            feat_num = (feat_num - means_pt) / stds_pt

            preds = encoder(
                feat_num,
                feat_species,
                feat_habitat,
            )

            paramsl = preds[:, [0, 1, 2, 3, 4, 5]]
            paramsm = preds[:, [6, 7, 8, 9, 10, 11]]
            paramsu = preds[:, [12, 13, 14, 15, 16, 17]]

            paramsl = rectify_parameters(paramsl)
            paramsm = rectify_parameters(paramsm)
            paramsu = rectify_parameters(paramsu)

            feat_grp["params_lower"][slc, :] = paramsl.detach().cpu().numpy()
            feat_grp["params_median"][slc, :] = paramsm.detach().cpu().numpy()
            feat_grp["params_upper"][slc, :] = paramsu.detach().cpu().numpy()

            ndvi_lower = double_logistic_function(t, paramsl).detach().cpu().numpy()
            ndvi_median = double_logistic_function(t, paramsm).detach().cpu().numpy()
            ndvi_upper = double_logistic_function(t, paramsu).detach().cpu().numpy()

            ndvi = ndvi.detach().cpu().numpy()
            ndsi = ndsi.detach().cpu().numpy()

            # Create masks for invalid, masked, outlier, NaN, and snow pixels
            is_unavailable = ndvi == INVALID
            is_masked = ndvi == NO_COVERAGE
            is_outlier = (
                ((ndvi > 10000) | (ndvi < -1000)) & ~is_unavailable & ~is_masked
            )
            is_nan = np.isnan(ndvi)
            is_snow = (ndsi >= 4300) & (ndsi <= 10000)

            valid_mask = ~(is_unavailable | is_masked | is_outlier | is_nan) | is_snow
            print("Valid pixels:", np.sum(valid_mask), "out of", valid_mask.size)

            ndvi = ndvi / 10000.0

            print(
                "NDVI stats:",
                np.nanmin(ndvi),
                np.nanmax(ndvi),
                np.nanmean(ndvi),
                np.nanmedian(ndvi),
            )
            print(
                "Predicted NDVI Lower stats:",
                np.nanmin(ndvi_lower),
                np.nanmax(ndvi_lower),
                np.nanmean(ndvi_lower),
                np.nanmedian(ndvi_lower),
            )
            print(
                "Predicted NDVI Median stats:",
                np.nanmin(ndvi_median),
                np.nanmax(ndvi_median),
                np.nanmean(ndvi_median),
                np.nanmedian(ndvi_median),
            )
            print(
                "Predicted NDVI Upper stats:",
                np.nanmin(ndvi_upper),
                np.nanmax(ndvi_upper),
                np.nanmean(ndvi_upper),
                np.nanmedian(ndvi_upper),
            )

            feat_grp_preds["ndvi_pred_lower"][slc, :] = (
                (ndvi_lower * 10000.0).round().astype(np.int16)
            )
            feat_grp_preds["ndvi_pred_median"][slc, :] = (
                (ndvi_median * 10000.0).round().astype(np.int16)
            )
            feat_grp_preds["ndvi_pred_upper"][slc, :] = (
                (ndvi_upper * 10000.0).round().astype(np.int16)
            )

            iqr = ndvi_upper - ndvi_lower

            print(
                "IQR stats:",
                np.nanmin(iqr),
                np.nanmax(iqr),
                np.nanmean(iqr),
                np.nanmedian(iqr),
            )

            # Define thresholds for anomalies
            lower_thresh = ndvi_lower - 1.5 * iqr
            upper_thresh = ndvi_upper + 1.5 * iqr

            # Compute anomalies only for valid NDVI
            is_lower_anomaly = (ndvi < lower_thresh) & valid_mask
            is_upper_anomaly = (ndvi > upper_thresh) & valid_mask

            print("Lower anomalies:", np.sum(is_lower_anomaly))
            print("Upper anomalies:", np.sum(is_upper_anomaly))

            # Initialize output
            anomalies = np.zeros_like(ndvi, dtype=np.int8)
            anomalies[is_lower_anomaly] = -1
            anomalies[is_upper_anomaly] = 1
            anomalies[is_masked | is_outlier | is_nan] = -128 | is_snow
            anomalies[is_unavailable] = 127

            print("Anomaly distribution:", np.unique(anomalies, return_counts=True))
            neg_mask = anomalies == -1
            pos_mask = anomalies == 1
            total_neg = np.sum(neg_mask)
            total_pos = np.sum(pos_mask)
            print(
                "Percent negative anomalies:",
                total_neg / valid_mask.sum() * 100.0 if valid_mask.sum() > 0 else 0.0,
            )
            print(
                "Percent positive anomalies:",
                total_pos / valid_mask.sum() * 100.0 if valid_mask.sum() > 0 else 0.0,
            )

            root_params["anomalies"][slc, :] = anomalies

            score = np.zeros_like(ndvi, dtype=np.float32)

            # Compute anomaly scores as the standardized distance from the predicted bounds, only for valid pixels
            below = (ndvi < ndvi_lower) & valid_mask
            above = (ndvi > ndvi_upper) & valid_mask
            score[below] = -(ndvi_lower[below] - ndvi[below]) / iqr[below]
            score[above] = (ndvi[above] - ndvi_upper[above]) / iqr[above]
            score[is_masked | is_outlier | is_nan] = np.nan | is_snow
            score[is_unavailable] = np.nan

            root_params["anomaly_scores"][slc, :] = score

            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument(
        "--encoder_path", type=str, default="../../checkpoints/encoder.pt"
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--features",
        type=str,
        default="dem,slope,easting,northing,twi,tri,mean_curv,profile_curv,plan_curv,roughness,median_forest_height,forest_mix_rate,tree_species,habitat",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Running inference...")
    print("Using encoder:", args.encoder_path)
    print("Output path:", args.output_path)
    print("Data path:", args.data_path)
    print("Using features:", args.features)
    print("Using device:", args.device)

    inference(args)

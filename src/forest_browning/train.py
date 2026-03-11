"""This script trains a model to predict NDVI time series for forest browning monitoring using a double logistic function parameterized by 6 parameters. The model is trained using quantile regression with a pinball loss, and includes constraints for periodicity and non-crossing of quantiles."""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import d2_pinball_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from forest_browning.dataset import MEANS, STDS, ZarrDataset
from forest_browning.mlp import MLPWithEmbeddings


DEFAULT_NUM_EPOCHS = 20
DEFAULT_LR = 0.005
DEFAULT_LR_DECAY_RATE = 0.01
DEFAULT_BATCH_SIZE = 1024
DEFAULT_DEVICE = "cuda"


def double_logistic_function(t, params):
    """Double logistic function to model NDVI time series, parameterized by 6 parameters.

    Args:
        t (torch.Tensor): Time values, shape (1, T).
        params (torch.Tensor): Parameters for the double logistic function, shape (batch_size, 6).

    Returns:
        torch.Tensor: Predicted NDVI values, shape (batch_size, T).
    """
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(params, 1, dim=1)
    # Apply softplus to ensure positivity of the parameters that need to be positive
    mat_minus_sos = nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = nn.functional.softplus(eos_minus_sen)
    # Define the double logistic function using the parameters
    sigmoid_sos_mat = nn.functional.sigmoid(
        -2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-10)
    )
    sigmoid_sen_eos = nn.functional.sigmoid(
        -2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-10)
    )
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m


def objective_pinball(params, t, ndvi, nan_mask, alpha=0.5, weights=None):
    """Pinball loss for quantile regression.

    Args:
        params (torch.Tensor): Predicted parameters for the double logistic function, shape (batch_size, 6).
        t (torch.Tensor): Time values, shape (1, T).
        ndvi (torch.Tensor): Observed NDVI values, shape (batch_size, T).
        nan_mask (torch.Tensor): Mask for NaN values, shape (batch_size, T).
        alpha (float, optional): Quantile level. Defaults to 0.5.
        weights (torch.Tensor, optional): Weights for each sample. Defaults to None.

    Returns:
        torch.Tensor: The computed pinball loss.
    """
    ndvi_pred = double_logistic_function(t, params)
    diff = ndvi - ndvi_pred
    loss = torch.max(torch.mul(alpha, diff), torch.mul((alpha - 1), diff))
    # Reweight the quantiles to prevent a degenerate solution
    if weights is not None:
        loss = loss * weights.unsqueeze(0)
    return torch.mean(loss[~nan_mask])


def train(args):
    """Train the model.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    torch.manual_seed(1)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"{args.output_dir}/runs/{run_id}")
    checkpoint_dir = f"{args.output_dir}/checkpoints/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Writing logs to {checkpoint_dir}")
    print(f"Run directory: {writer.log_dir}")

    print("Loading dataset...")

    ds = ZarrDataset(
        args.data_path,
        args.features,
        batch_size=args.batch_size,
        chunk_size=8192,
        shuffle_chunks=True,
        seed=42,
    )
    missingness = ds.missingness
    missingness = torch.from_numpy(missingness).to(args.device)
    t = torch.from_numpy(ds.t).float().to(args.device)

    if isinstance(args.features, str):
        args.features = args.features.split(",")
    print("Using features: {}".format(ds.features))

    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    means_pt = torch.tensor([MEANS[f] for f in ds.num_features]).unsqueeze(0)
    stds_pt = torch.tensor([STDS[f] for f in ds.num_features]).unsqueeze(0)

    # this model has ~475k parameters
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

    print(
        "Number of parameters: {}".format(
            sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        )
    )

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)

    # Create a fixed time grid for evaluating the non-crossing constraints
    NC_GRID_SIZE = 32
    t_grid = torch.linspace(0, 1.0, NC_GRID_SIZE, device=args.device).unsqueeze(0)

    print("Starting training...")

    n_iterations = 0
    total_iterations = args.num_epochs * ds.n_batches
    for epoch in range(args.num_epochs):
        ds.set_epoch(epoch)
        print(f"Starting epoch {epoch + 1}")
        for sample in tqdm(loader, total=len(loader)):
            ndvi, feat = sample

            # Create mask for NaN and outlier values in NDVI, and map NDVI from int16 to float32 in the range [-0.1, 1.0]
            nan_mask = torch.isnan(ndvi) | (ndvi == -(2**15)) | (ndvi == 2**15 - 1)
            ndvi = ndvi.float() / 10000.0
            outlier_mask = (ndvi > 1) | (ndvi < -0.1)
            nan_mask = nan_mask | outlier_mask

            feat_num = feat[:, ds.num_feature_indices]
            feat_species = feat[:, ds.mapping_features["tree_species"]].int()
            feat_habitat = feat[:, ds.mapping_features["habitat"]].int()
            # Map invalid species code 255 to 16 (indicating missing)
            feat_species[feat_species == 255] = 16

            # Standardize input
            feat_num = (feat_num - means_pt) / stds_pt

            feat_num = feat_num.to(args.device, non_blocking=True)
            feat_species = feat_species.to(args.device, non_blocking=True)
            feat_habitat = feat_habitat.to(args.device, non_blocking=True)

            t_ndvi_train = ndvi.float().to(args.device, non_blocking=True)
            t_nan_mask_train = nan_mask.to(args.device, non_blocking=True)

            # Predict quantiles of parameters for the double logistic function
            preds = encoder(
                feat_num,
                feat_species,
                feat_habitat,
            )

            paramsl = preds[:, [0, 1, 2, 3, 4, 5]]
            paramsm = preds[:, [6, 7, 8, 9, 10, 11]]
            paramsu = preds[:, [12, 13, 14, 15, 16, 17]]

            lossl = objective_pinball(
                paramsl,
                t,
                t_ndvi_train,
                t_nan_mask_train,
                alpha=0.25,
                weights=missingness,
            )
            lossm = objective_pinball(
                paramsm,
                t,
                t_ndvi_train,
                t_nan_mask_train,
                alpha=0.50,
                weights=missingness,
            )
            lossu = objective_pinball(
                paramsu,
                t,
                t_ndvi_train,
                t_nan_mask_train,
                alpha=0.75,
                weights=missingness,
            )

            # Add constraint to ensure periodicity
            t_start = torch.full((feat.shape[0], 1), 0, device=args.device)
            t_end = torch.full((feat.shape[0], 1), 1, device=args.device)
            startl = double_logistic_function(t_start, paramsl)
            endl = double_logistic_function(t_end, paramsl)
            periodic_loss_l = torch.mean((startl - endl) ** 2)
            startm = double_logistic_function(t_start, paramsm)
            endm = double_logistic_function(t_end, paramsm)
            periodic_loss_m = torch.mean((startm - endm) ** 2)
            startu = double_logistic_function(t_start, paramsu)
            endu = double_logistic_function(t_end, paramsu)
            periodic_loss_u = torch.mean((startu - endu) ** 2)
            total_periodic_loss = periodic_loss_l + periodic_loss_m + periodic_loss_u

            lambda_periodic = 1

            # Add constraint to ensure non-crossing of quantiles
            t_grid_b = t_grid.repeat(paramsl.shape[0], 1)
            ndvi_lower_grid = double_logistic_function(t_grid_b, paramsl)
            ndvi_middle_grid = double_logistic_function(t_grid_b, paramsm)
            ndvi_upper_grid = double_logistic_function(t_grid_b, paramsu)
            violation_lu = torch.relu(ndvi_lower_grid - ndvi_upper_grid)
            violation_lm = torch.relu(ndvi_lower_grid - ndvi_middle_grid)
            violation_mu = torch.relu(ndvi_middle_grid - ndvi_upper_grid)
            violation = violation_lu + violation_lm + violation_mu
            per_sample_noncross = violation.mean(dim=1)
            total_noncross = per_sample_noncross.mean()

            lambda_nc = 10.0

            loss = (
                lossl
                + lossm
                + lossu
                + lambda_periodic * total_periodic_loss
                + lambda_nc * total_noncross
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate with exponential decay
            new_lrate = args.lr * (
                args.lr_decay_rate ** (n_iterations / total_iterations)
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lrate

            if (n_iterations + 1) % 10 == 0:
                writer.add_scalar("Loss/train", loss, n_iterations + 1)
                for pi, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(
                        "LearningRate[{}]".format(pi),
                        param_group["lr"],
                        n_iterations + 1,
                    )

            if (n_iterations + 1) % 100 == 0:
                with torch.no_grad():
                    # Plot the fitted curves for a random subset of samples, along with the observed NDVI values, and log to TensorBoard
                    fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharey=True)

                    t_fit = (
                        torch.linspace(0, 1, 1000)
                        .unsqueeze(0)
                        .repeat(paramsl.shape[0], 1)
                    )
                    ndvi_lower = double_logistic_function(t_fit, paramsl.cpu())
                    ndvi_middle = double_logistic_function(t_fit, paramsm.cpu())
                    ndvi_upper = double_logistic_function(t_fit, paramsu.cpu())

                    random_indices = np.random.choice(
                        np.arange(paramsl.shape[0]), size=4, replace=False
                    )
                    for pl_idx, bi in enumerate(random_indices):
                        row, col = divmod(pl_idx, 2)

                        masked_ndvi_train = ndvi[bi][~nan_mask[bi]]

                        # Convert fractional year (t) to day-of-year for plotting
                        doy_array = (ds.t * 365).astype(np.float32)
                        masked_doy_train = doy_array[~nan_mask[bi]]

                        ax[row, col].scatter(
                            masked_doy_train, masked_ndvi_train, label="Observed NDVI"
                        )
                        ax[row, col].fill_between(
                            (t_fit * 365)[bi],
                            ndvi_lower[bi],
                            ndvi_middle[bi],
                            alpha=0.2,
                            color="red",
                        )

                        ax[row, col].fill_between(
                            (t_fit * 365)[bi],
                            ndvi_middle[bi],
                            ndvi_upper[bi],
                            alpha=0.2,
                            color="green",
                        )
                        ax[row, col].set_xlim(0, 365)
                        ax[row, col].set_xlabel("Day of year")
                        ax[0, 0].set_ylabel("NDVI")
                        ax[1, 0].set_ylabel("NDVI")

                    writer.add_figure(f"Fit/iter_{n_iterations + 1}", fig, n_iterations)

                    # Compute D2 pinball score on the training batch for each quantile and log to TensorBoard
                    ndvi_lower = double_logistic_function(t.cpu(), paramsl.cpu())
                    ndvi_middle = double_logistic_function(t.cpu(), paramsm.cpu())
                    ndvi_upper = double_logistic_function(t.cpu(), paramsu.cpu())

                    all_masked_ndvi_lower = ndvi_lower[~nan_mask].cpu()
                    all_masked_ndvi_middle = ndvi_middle[~nan_mask].cpu()
                    all_masked_ndvi_upper = ndvi_upper[~nan_mask].cpu()
                    all_masked_ndvi_train = ndvi[~nan_mask].cpu()
                    d2_score_lower = d2_pinball_score(
                        all_masked_ndvi_train, all_masked_ndvi_lower, alpha=0.25
                    )
                    d2_score_middle = d2_pinball_score(
                        all_masked_ndvi_train, all_masked_ndvi_middle, alpha=0.50
                    )
                    d2_score_upper = d2_pinball_score(
                        all_masked_ndvi_train, all_masked_ndvi_upper, alpha=0.75
                    )

                    writer.add_scalar(
                        "D2PinballScoreLower/train", d2_score_lower, n_iterations
                    )
                    writer.add_scalar(
                        "D2PinballScoreMiddle/train", d2_score_middle, n_iterations
                    )
                    writer.add_scalar(
                        "D2PinballScoreUpper/train", d2_score_upper, n_iterations
                    )
            n_iterations += 1

        torch.save(
            encoder.state_dict(), f"{checkpoint_dir}/encoder_epoch{epoch + 1}.pt"
        )

    torch.save(encoder.state_dict(), f"{checkpoint_dir}/encoder.pt")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--features",
        type=str,
        default="dem,slope,easting,northing,twi,tri,mean_curv,profile_curv,plan_curv,roughness,median_forest_height,forest_mix_rate,tree_species,habitat",
    )
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lr_decay_rate", type=float, default=DEFAULT_LR_DECAY_RATE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    train(args)

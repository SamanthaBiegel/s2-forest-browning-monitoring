"""Defines the MLPWithEmbeddings model, which is a multilayer perceptron that takes in numerical features as well as categorical species and habitat information. The features are embedded and concatenated with the numerical features before being passed through the MLP. Optionally includes a skip connection."""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


class MLPWithEmbeddings(nn.Module):
    """Multilayer perceptron that takes in numerical features as well as categorical species and habitat information.

    The features are embedded and concatenated with the numerical features before being passed through the MLP. Optionally includes a skip connection.
    """

    def __init__(
        self,
        d_num: int,
        d_out: int | None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        skip_connection: bool,
        n_species: int,
        species_emb_dim: int,
        n_habitats: int,
        habitat_emb_dim: int,
    ) -> None:
        """Initializes the MLPWithEmbeddings model.

        Args:
            d_num (int): Dimensionality of the numerical features.
            d_out (int | None): Dimensionality of the output layer.
            n_blocks (int): Number of blocks in the MLP.
            d_block (int): Dimensionality of each block in the MLP.
            dropout (float): Dropout probability.
            skip_connection (bool): Whether to use skip connections.
            n_species (int): Number of unique species.
            species_emb_dim (int): Dimensionality of the species embedding.
            n_habitats (int): Number of unique habitats.
            habitat_emb_dim (int): Dimensionality of the habitat embedding.

        Raises:
            ValueError: If the number of blocks is not positive or if skip connection is used without enough blocks.
        """
        super().__init__()

        self.species_emb = nn.Embedding(n_species, species_emb_dim)
        self.habitat_emb = nn.Embedding(n_habitats, habitat_emb_dim)

        d_in_total = d_num + species_emb_dim + habitat_emb_dim

        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, however: {n_blocks=}")

        if skip_connection:
            assert n_blocks > 1

        blocks = []
        for i in range(n_blocks):
            if i == 0:
                # First layer takes in the concatenated numerical features and embeddings
                blocks.append(
                    _named_sequential(
                        ("linear", nn.Linear(d_in_total, d_block)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout(dropout)),
                    )
                )
            elif skip_connection and (i == n_blocks // 2):
                # Use skip connection by concatenating the original input to the current layer input
                blocks.append(
                    _named_sequential(
                        ("linear", nn.Linear(d_block + d_in_total, d_block)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout(dropout)),
                    )
                )
            else:
                # Subsequent layers take in the output of the previous layer
                blocks.append(
                    _named_sequential(
                        ("linear", nn.Linear(d_block, d_block)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout(dropout)),
                    )
                )

        self.blocks = nn.ModuleList(blocks)
        self.output = None if d_out is None else nn.Linear(d_block, d_out)
        self.skip_connection = skip_connection

    def forward(
        self,
        x_num: Tensor,
        species_idx: Tensor,
        habitat_freqs: Tensor,
    ) -> Tensor:
        """Forward pass of the MLPWithEmbeddings model.

        Args:
            x_num (Tensor): Tensor of shape (batch_size, d_num) containing the numerical features.
            species_idx (Tensor): Tensor of shape (batch_size,) containing the species indices.
            habitat_freqs (Tensor): Tensor of shape (batch_size, n_habitats) containing the habitat frequencies.

        Returns:
            Tensor: Tensor of shape (batch_size, d_out) containing the output features.
        """
        species_vec = self.species_emb(species_idx).squeeze(1)
        # Compute habitat frequency weights
        habitat_weights = habitat_freqs / (
            habitat_freqs.sum(dim=1, keepdim=True) + 1e-8
        )
        # Compute weighted average of habitat embeddings
        habitat_vec = habitat_weights @ self.habitat_emb.weight

        x = torch.cat([x_num, species_vec, habitat_vec], dim=-1)

        out = x.clone()
        for i, block in enumerate(self.blocks):
            if self.skip_connection and (i == len(self.blocks) // 2):
                out = torch.cat([x, out], axis=-1)
            out = block(out)

        if self.output is not None:
            out = self.output(out)
        return out

from typing import List

import torch
import torch.nn as nn

from attrackt.autoencoder.decoder3d import Decoder3d
from attrackt.autoencoder.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)
from attrackt.autoencoder.encoder3d import Encoder3d


class AutoencoderModel3d(nn.Module):
    def __init__(
        self,
        num_in_out_channels: int,
        num_intermediate_channels: int,
        num_z_channels: int,
        attention_resolutions: List[int],
        num_res_blocks: int,
        resolution: int,
        channels_multiply_factor: List[int],
        embedding_dimension: int,
        device: str = "cuda",
        variational: bool = False,
    ):
        super(AutoencoderModel3d, self).__init__()
        encoder = Encoder3d(
            num_in_out_channels=num_in_out_channels,
            num_intermediate_channels=num_intermediate_channels,
            num_z_channels=num_z_channels if not variational else 2 * num_z_channels,
            attention_resolutions=attention_resolutions,
            num_res_blocks=num_res_blocks,
            start_resolution=resolution,
            channels_multiply_factor=channels_multiply_factor,
        )

        decoder = Decoder3d(
            num_in_out_channels=num_in_out_channels,
            num_intermediate_channels=num_intermediate_channels,
            channels_multiply_factor=channels_multiply_factor,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            end_resolution=resolution,
            num_z_channels=num_z_channels,
        )

        self.device = device
        self.pre_quant_conv = torch.nn.Conv3d(
            2 * num_z_channels, 2 * embedding_dimension, 1
        )
        self.post_quant_conv = torch.nn.Conv3d(embedding_dimension, num_z_channels, 1)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.pre_quant_conv = self.pre_quant_conv.to(self.device)
        self.post_quant_conv = self.post_quant_conv.to(self.device)
        self.variational = variational

    def forward(
        self,
        im: torch.Tensor,
        only_encode: bool = False,
    ) -> torch.Tensor:
        """forward.
        This function takes in a torch tensor ( B, C, D, H, W).
        Generates an embedding of the image using the encoder. (B, z, d, h, w)
        Returns the output of the decoder.

        Parameters
        ----------
        im : torch.Tensor
            im
        only_encode : bool
            If set to True, only the encoded embedding is returned.
        """

        if self.variational:
            hidden = self.encoder(im)  # B 2z d h w
            moments = self.pre_quant_conv(hidden)  # B 2e d h w
            posterior = DiagonalGaussianDistribution(parameters=moments)
            if only_encode:
                posterior_mean = posterior.get_mean()  # B e d h w
                posterior_mean = posterior_mean.view(posterior_mean.shape[0], -1)
                return posterior_mean  # B edhw
            else:
                z = posterior.rsample()  # B e d h w
                y = self.decoder(self.post_quant_conv(z))  # B C D H W
                return y, posterior
        else:
            embeddings = self.encoder(im)  # B z d h w
            if only_encode:
                embeddings = embeddings.view(embeddings.shape[0], -1)
                return embeddings  # B zdhw
            else:
                y = self.decoder(embeddings)  # B C D H W
                return y

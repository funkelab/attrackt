from typing import List, Tuple

import torch
import torch.nn as nn

from attrackt.autoencoder.resnet3d import ResnetBlock3d
from attrackt.autoencoder.utils import (
    Upsample3d,
    make_attention,
    nonlinearity,
    normalize,
)


class Decoder3d(nn.Module):
    def __init__(
        self,
        num_in_out_channels: int,
        num_intermediate_channels: int,
        channels_multiply_factor: Tuple[int, ...],
        num_res_blocks: int,
        attention_resolutions: List[int],
        end_resolution: int,
        num_z_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.num_intermediate_channels = num_intermediate_channels
        self.num_resolutions = len(channels_multiply_factor)
        self.num_res_blocks = num_res_blocks
        self.end_resolution = end_resolution
        self.num_z_channels = num_z_channels
        num_channels_in_level = (
            num_intermediate_channels
            * channels_multiply_factor[self.num_resolutions - 1]
        )
        current_resolution = end_resolution // 2 ** (self.num_resolutions - 1)
        self.start_resolution = current_resolution

        self.conv_in = torch.nn.Conv3d(
            in_channels=num_z_channels,
            out_channels=num_channels_in_level,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="zeros",
        )

        self.mid_module = nn.Module()
        self.mid_module.block_1 = ResnetBlock3d(
            num_in_channels=num_channels_in_level,
            num_out_channels=num_channels_in_level,
        )

        self.mid_module.attention_1 = make_attention(
            num_channels_in_level,
            num_spatial_dims=3,
        )

        self.mid_module.block_2 = ResnetBlock3d(
            num_in_channels=num_channels_in_level,
            num_out_channels=num_channels_in_level,
        )

        # upsampling
        self.up_module_list = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attention = nn.ModuleList()
            num_channels_out_level = (
                num_intermediate_channels * channels_multiply_factor[i_level]
            )
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3d(
                        num_in_channels=num_channels_in_level,
                        num_out_channels=num_channels_out_level,
                    )
                )
                num_channels_in_level = num_channels_out_level
                if current_resolution in attention_resolutions:
                    attention.append(
                        make_attention(
                            num_channels_in_level,
                            num_spatial_dims=3,
                        )
                    )
            up_module = nn.Module()
            up_module.block = block
            up_module.attention = attention
            if i_level != 0:
                up_module.upsample = Upsample3d(num_channels_in_level)
                current_resolution = current_resolution * 2
            self.up_module_list.insert(0, up_module)  # prepend to get consistent order

        self.norm_out = normalize(num_channels_in_level)
        self.conv_out = torch.nn.Conv3d(
            num_channels_in_level,
            num_in_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_module.block_1(h)
        h = self.mid_module.attention_1(h)
        h = self.mid_module.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up_module_list[i_level].block[i_block](h)
                if len(self.up_module_list[i_level].attention) > 0:
                    h = self.up_module_list[i_level].attention[i_block](h)
            if i_level != 0:
                h = self.up_module_list[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

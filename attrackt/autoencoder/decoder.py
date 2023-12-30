from typing import List, Tuple

import torch
import torch.nn as nn

from attrackt.autoencoder.resnet import ResnetBlock
from attrackt.autoencoder.utils import Upsample, make_attention, nonlinearity, normalize


class Decoder(nn.Module):
    def __init__(
        self,
        num_in_out_channels: int,
        num_intermediate_channels: int,
        channels_multiply_factor: Tuple[int, ...],
        num_res_blocks: int,
        attention_resolutions: List[int],
        end_resolution: int,
        num_z_channels: int,
        attention_type: str = "vanilla",
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """__init__.

        Parameters
        ----------
        num_in_out_channels : int
            num_in_out_channels is the number of channels output by the decoder.
            It is also the number of channels in the input to the encoder (see :class:`attrackt.models.encoder.Encoder` for details).
        num_intermediate_channels : int
            num_intermediate_channels is the number of channels in the intermediate layers of the decoder.
        channels_multiply_factor : Tuple[int, ...]
            channels_multiply_factor is the factor by which the number of channels in the intermediate layers of the decoder is multiplied at each resolution level.
        num_res_blocks : int
            num_res_blocks is the number of residual blocks in each resolution level of the decoder.
        attention_resolutions : List[int]
            attention_resolutions is a list of resolutions at which attention is applied in the decoder.
        end_resolution : int
            end_resolution is the resolution of the output of the decoder.
            For example, if the input tensor shape to the Encoder was 8 x 1 x 64 x 64, then the end_resolution should be 64.
        num_z_channels : int
            num_z_channels is the number of channels in the input tensor to the decoder.
        attention_type : str
            attention_type is the type of attention to be used in the decoder.
        kernel_size : int
            kernel_size is the size of the kernel in the convolutional layers of the decoder.
        stride : int
            stride is the stride in the convolutional layers of the decoder.
        padding : int
            padding is the padding in the convolutional layers of the decoder.
        """
        super().__init__()
        self.num_intermediate_channels = num_intermediate_channels
        self.num_resolutions = len(channels_multiply_factor)
        self.num_res_blocks = num_res_blocks
        self.end_resolution = end_resolution

        num_channels_in_level = (
            num_intermediate_channels
            * channels_multiply_factor[self.num_resolutions - 1]
        )
        current_resolution = end_resolution // 2 ** (self.num_resolutions - 1)
        self.start_resolution = current_resolution

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            in_channels=num_z_channels,
            out_channels=num_channels_in_level,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="zeros",
        )

        # middle
        self.mid_module = nn.Module()
        self.mid_module.block_1 = ResnetBlock(
            num_in_channels=num_channels_in_level,
            num_out_channels=num_channels_in_level,
        )

        self.mid_module.attention_1 = make_attention(
            num_channels_in_level,
            # attention_type=attention_type
        )
        self.mid_module.block_2 = ResnetBlock(
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
                    ResnetBlock(
                        num_in_channels=num_channels_in_level,
                        num_out_channels=num_channels_out_level,
                    )
                )
                num_channels_in_level = num_channels_out_level
                if current_resolution in attention_resolutions:
                    attention.append(
                        make_attention(
                            num_channels_in_level,
                            # attention_type=attention_type
                        )
                    )
            up_module = nn.Module()
            up_module.block = block
            up_module.attention = attention
            if i_level != 0:
                up_module.upsample = Upsample(num_channels_in_level)
                current_resolution = current_resolution * 2
            self.up_module_list.insert(0, up_module)  # prepend to get consistent order

        # end
        self.norm_out = normalize(num_channels_in_level)
        self.conv_out = torch.nn.Conv2d(
            num_channels_in_level,
            num_in_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid_module.block_1(h)
        h = self.mid_module.attention_1(h)
        h = self.mid_module.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up_module_list[i_level].block[i_block](h)
                if len(self.up_module_list[i_level].attention) > 0:
                    h = self.up_module_list[i_level].attention[i_block](h)
            if i_level != 0:
                h = self.up_module_list[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

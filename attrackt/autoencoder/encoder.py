from typing import List, Tuple

import torch
import torch.nn as nn

from attrackt.autoencoder.resnet import ResnetBlock
from attrackt.autoencoder.utils import (
    Downsample,
    make_attention,
    nonlinearity,
    normalize,
)


class Encoder(nn.Module):
    def __init__(
        self,
        num_in_out_channels: int,
        num_intermediate_channels: int,
        num_z_channels: int,
        num_res_blocks: int,
        channels_multiply_factor: Tuple[int],
        start_resolution: int,
        attention_resolutions: List[int],
        attention_type: str = "vanilla",
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """__init__.

        Parameters
        ----------
        num_in_out_channels : int
            num_in_out_channels is the number of channels in the input tensor.
            It is also the number of channels in the tensor output from the `Decoder` (see :class:`attrackt.models.decoder.Decoder` for details.)
        num_intermediate_channels : int
            num_intermediate_channels is the number of channels in an intermediate produced tensor.
            Look at `conv_in` in the code below.
        num_z_channels : int
            num_z_channels is the number of channels in the tensor output from the encoder.
        num_res_blocks : int
            num_res_blocks is the number of residual blocks in the encoder.
        channels_multiply_factor : Tuple[int]
            channels_multiply_factor is a tuple of integers that are used to multiply the number of channels in the intermediate tensor.
        start_resolution : int
            start_resolution is the resolution of the input tensor.
            For example, if a tensor of shape [8, 1, 64, 64] is passed to the encoder, then the `start_resolution` is 64.
        attention_resolutions : List[int]
            attention_resolutions is a list of integers that specify the resolutions at which attention is applied.
        attention_type : str
            attention_type is the type of attention applied.
        kernel_size : int
            kernel_size is the size of the kernel used in the convolutional layers.
        stride : int
            stride is the stride used in the convolutional layers.
        padding : int
            padding is the padding used in the convolutional layers.
        """
        super().__init__()
        self.num_intermediate_channels = num_intermediate_channels
        self.num_resolutions = len(channels_multiply_factor)
        self.num_res_blocks = num_res_blocks
        self.start_resolution = start_resolution
        self.num_in_channels = num_in_out_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels=self.num_in_channels,
            out_channels=self.num_intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="zeros",
        )

        current_resolution = self.start_resolution  # for example, 64
        in_channels_multiply_factor = (1,) + tuple(
            channels_multiply_factor
        )  # (1,1,2,2,4)
        self.down_module_list = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attention = nn.ModuleList()
            num_in_channels_level = (
                self.num_intermediate_channels * in_channels_multiply_factor[i_level]
            )
            num_out_channels_level = (
                self.num_intermediate_channels * channels_multiply_factor[i_level]
            )
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        num_in_channels=num_in_channels_level,
                        num_out_channels=num_out_channels_level,
                    )
                )
                num_in_channels_level = num_out_channels_level
                if current_resolution in attention_resolutions:
                    attention.append(
                        make_attention(
                            in_channels=num_in_channels_level,
                            # attention_type=attention_type,
                        )
                    )
            down_module = nn.Module()
            down_module.block = block
            down_module.attention = attention
            if i_level != self.num_resolutions - 1:
                down_module.downsample = Downsample(num_in_channels_level)
                current_resolution = current_resolution // 2
            self.down_module_list.append(down_module)

        # middle
        self.mid_module = nn.Module()
        self.mid_module.block_1 = ResnetBlock(
            num_in_channels=num_in_channels_level,
            num_out_channels=num_in_channels_level,
        )
        self.mid_module.attention_1 = make_attention(
            in_channels=num_in_channels_level,
            # attention_type=attention_type
        )
        self.mid_module.block_2 = ResnetBlock(
            num_in_channels=num_in_channels_level,
            num_out_channels=num_in_channels_level,
        )

        # end
        self.normalize_out = normalize(num_in_channels_level)
        self.convolution_out = torch.nn.Conv2d(
            num_in_channels_level,
            num_z_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        """
        x : torch.Tensor(8,1,64,64)
        """
        # downsampling
        hs = [self.conv_in(x)]  # (8,128,64,64)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down_module_list[i_level].block[i_block](hs[-1])
                if len(self.down_module_list[i_level].attention) > 0:
                    h = self.down_module_list[i_level].attention[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down_module_list[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid_module.block_1(h)
        h = self.mid_module.attention_1(h)
        h = self.mid_module.block_2(h)

        # end
        h = self.normalize_out(h)
        h = nonlinearity(h)
        h = self.convolution_out(h)  # (8,4,4,4)

        return h

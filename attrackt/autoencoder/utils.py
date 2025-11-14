from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Downsample3d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="trilinear")
        x = self.conv(x)
        return x


class Upsample3d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


def normalize(in_channels: int, num_groups: int = 16) -> torch.nn.GroupNorm:
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.normalize = normalize(in_channels)

        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.projection_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.normalize(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.projection_out(h_)

        return x + h_


class AttentionBlock3d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.normalize = normalize(in_channels)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.projection_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.normalize(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, d, h, w = q.shape
        q = q.reshape(b, c, d * h * w)
        q = q.permute(0, 2, 1)  # b,dhw,c
        k = k.reshape(b, c, d * h * w)  # b,c,dhw
        w_ = torch.bmm(q, k)  # b,dhw,dhw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, d * h * w)
        w_ = w_.permute(0, 2, 1)  # b,dhw,dhw (first dhw of k, second of q)
        h_ = torch.bmm(
            v, w_
        )  # b, c,dhw (dhw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, d, h, w)

        h_ = self.projection_out(h_)

        return x + h_


def make_attention(
    in_channels: int, num_spatial_dims: int = 2
) -> Union["AttentionBlock", "AttentionBlock3d"]:
    if num_spatial_dims == 2:
        return AttentionBlock(in_channels)
    elif num_spatial_dims == 3:
        return AttentionBlock3d(in_channels)
    else:
        raise ValueError(f"Unsupported num_spatial_dims: {num_spatial_dims}")


def update_corners(
    tly: int,
    tlx: int,
    bry: int,
    brx: int,
    crop_size: Tuple[int, ...],
    tlz: Optional[int] = None,
    brz: Optional[int] = None,
) -> Union[Tuple[int, int, int, int], Tuple[int, int, int, int, int, int]]:
    if len(crop_size) == 2:
        tly += crop_size[0] // 2
        bry += crop_size[0] // 2
        tlx += crop_size[1] // 2
        brx += crop_size[1] // 2
        return tly, tlx, bry, brx
    elif len(crop_size) == 3:
        if tlz is None or brz is None:
            raise ValueError("tlz and brz must be provided for 3D cropping.")
        tlz += crop_size[0] // 2
        brz += crop_size[0] // 2
        tly += crop_size[1] // 2
        bry += crop_size[1] // 2
        tlx += crop_size[2] // 2
        brx += crop_size[2] // 2
        return tlz, tly, tlx, brz, bry, brx
    else:
        raise ValueError(f"Invalid crop_size length: {len(crop_size)}")


def get_corners_bbox(
    position: np.ndarray, crop_size: Tuple[int, ...]
) -> Tuple[int, ...]:
    if len(position) == 2 and len(crop_size) == 2:
        tly = int(position[0] - crop_size[0] // 2)
        tlx = int(position[1] - crop_size[1] // 2)
        bry = tly + crop_size[0]
        brx = tlx + crop_size[1]
        return tly, tlx, bry, brx
    elif len(position) == 3 and len(crop_size) == 3:
        tlz = int(position[0] - crop_size[0] // 2)
        tly = int(position[1] - crop_size[1] // 2)
        tlx = int(position[2] - crop_size[2] // 2)
        brz = tlz + crop_size[0]
        bry = tly + crop_size[1]
        brx = tlx + crop_size[2]
        return tlz, tly, tlx, brz, bry, brx
    else:
        raise ValueError(
            f"Unsupported input shapes: position={position.shape}, crop_size={crop_size}"
        )

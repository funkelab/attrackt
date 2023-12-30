from typing import Tuple

import numpy as np


def update_corners(tly, tlx, bry, brx, crop_size, tlz=None, brz =None):
    if len(crop_size) == 2:
        tly += crop_size[0] // 2
        bry += crop_size[0] // 2
        tlx += crop_size[1] // 2
        brx += crop_size[1] // 2
        return tly, tlx, bry, brx
    elif len(crop_size) == 3:
        tlz += crop_size[0] // 2
        brz += crop_size[0] // 2
        tly += crop_size[1] // 2
        bry += crop_size[1] // 2
        tlx += crop_size[2] // 2
        brx += crop_size[2] // 2
        return tlz, tly, tlx, brz, bry, brx

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


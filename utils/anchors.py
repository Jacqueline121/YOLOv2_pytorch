import torch
import numpy as np
from config.config import cfg


def generate_anchors():
    anchors = cfg.ANCHORS
    anchor_num = len(anchors)
    anchor_dim = len(anchors[0])

    if anchor_dim == 4:
        anchors[:, :2] = 0
    else:
        anchors = np.cat([np.zeros_like(anchors), anchors], 1)

    stride = cfg.REDUCTION

    height, width = cfg.TEST_SIZE / stride

    A = anchor_num
    K = height * width
    shift_x = np.arange(0, width)
    shift_y = np.arange(0, height)
    shifts_x, shifts_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shifts_x.ravel(), shifts_y.ravel(), shifts_x.ravel(), shifts_y.ravel())).transpose()

    all_anchors = anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    all_anchors = all_anchors.reshape(-1, 4)

    all_anchors = torch.from_numpy(all_anchors)

    return all_anchors
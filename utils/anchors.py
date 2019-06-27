import torch
import numpy as np
from config.config import cfg


def generate_anchors():
    anchors = torch.FloatTensor(cfg.ANCHORS)
    anchor_num = anchors.size(0)
    height = int(cfg.INPUT_SIZE[0] / cfg.STRIDE)
    width = int(cfg.INPUT_SIZE[1] / cfg.STRIDE)

    A = anchor_num
    K = height * width

    shift_x = torch.arange(0, height)
    shift_y = torch.arange(0, width)
    shifts_x, shifts_y = torch.meshgrid([shift_x, shift_y])

    ctrs_x = shifts_x.t().contiguous().float()
    ctrs_y = shifts_y.t().contiguous().float()

    ctrs = torch.cat([ctrs_x.view(-1, 1), ctrs_y.view(-1, 1)], dim=-1)

    all_anchors = torch.cat([ctrs.view(K, 1, 2).expand(K, A, 2),
                             anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

    all_anchors = all_anchors.contiguous().view(K * A, 4)

    return all_anchors


if __name__ == '__main__':
    anchor = generate_anchors()
    print(anchor[:12, :])
    print('nn')



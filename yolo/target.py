import torch
import torch.nn as nn
import numpy as np
from config.config import cfg
from utils.bbox_operation import bbox_overlaps


def yolo_loss(output, ground_truth, height, width):
    """
    calculate the target value for the predicted bounding box
    :param output: output of the darknet, size:[B, C, H, W]
    :param gt: ground truth, size:[B, N, 5], (x1, y1, x2, y2, o)
    :return: masks and targets


    where:
    B: is the batch size, equal to 10
    C: is the channel of the output, equal to num_anchor*(5 + num_class)
    H: is the height of the output, equal to 7
    W: is the width of the output, equal to 7
    N: is the number of ground truth boxes
    """

    # define parameters
    anchors = cfg.ANCHORS
    reduction = cfg.REDUCTION
    coord_scale = cfg.COORD_SCALE
    noobject_scale = cfg.NO_OBJECT_SCALE
    object_scale = cfg.OBJECT_SCALE
    class_scale = cfg.CLASS_SCALE
    thresh = cfg.THRESH
    class_num = cfg.CLASS_NUM

    anchor_num = len(anchors)
    anchor_dim = len(anchors[0])
    anchors = torch.Tensor(anchors)

    if anchor_dim == 4:
        anchors[:, :2] = 0
    else:
        anchors = torch.cat([torch.zeros_like(anchors), anchors], 1)

    coord = output[0]
    conf = output[1]
    cls = output[2]

    gt_boxes_batch = ground_truth[0]
    gt_classes_batch = ground_truth[1]
    num_obj_batch = ground_truth[2]

    batch_size = conf.size(0)

    # calculate the coordinates of the predicted boxes
    pred_boxes = torch.FloatTensor(batch_size * anchor_num * height * width, 4)

    ctr_x = torch.range(0, width - 1).repeat(height, 1).contiguous().view(width * height)
    ctr_y = torch.range(0, height - 1).repeat(width, 1).t().contiguous().view(width * height)
    anchor_w = anchors[:, 2].contiguous().view(anchor_num, 1)
    anchor_h = anchors[:, 3].contiguous().view(anchor_num, 1)

    if torch.cuda.is_available():
        pred_boxes = pred_boxes.cuda()
        ctr_x = ctr_x.cuda()
        ctr_y = ctr_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    pred_boxes[:, 0] = (coord[:, :, 0].detach() + ctr_x).view(-1)
    pred_boxes[:, 1] = (coord[:, :, 1].detach() + ctr_y).view(-1)
    pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
    pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)

    pred_boxes = pred_boxes.cpu()

    # build target
    coord_target = torch.zeros(batch_size, anchor_num, 4, height * width, requires_grad=False)
    conf_target = torch.zeros(batch_size, anchor_num, height * width, requires_grad=False)
    cls_target = torch.zeros(batch_size, anchor_num, height * width, requires_grad=False)

    coord_mask = torch.zeros(batch_size, anchor_num, 1, height * width, requires_grad=False)
    conf_mask = torch.ones(batch_size, anchor_num, height * width, requires_grad=False) * noobject_scale
    cls_mask = torch.zeros(batch_size, anchor_num, height * width, requires_grad=False).byte()

    for i in range(batch_size):
        if len(ground_truth[i]) == 0:
            continue

        cur_pred_boxes = pred_boxes[i * (anchor_num * height * width): (i + 1) * (anchor_num * height * width)]
        gt_boxes = torch.zeros(ground_truth[i].size(0), 4)
        for j, gt in enumerate(ground_truth[i]):
            gt_boxes[j, 0] = ((gt[0] + gt[2]) / 2) / reduction
            gt_boxes[j, 1] = ((gt[1] + gt[3]) / 2) / reduction
            gt_boxes[j, 2] = (gt[2] - gt[0]) / reduction
            gt_boxes[j, 3] = (gt[3] - gt[1]) / reduction

        gt_pred_iou = bbox_overlaps(gt_boxes, cur_pred_boxes)
        mask = (gt_pred_iou > thresh).sum(0) >= 1
        conf_mask[i][mask.view_as(conf_mask[i])] = 0

        gt_boxes_wh = gt_boxes.clone()
        gt_boxes_wh[:, :2] = 0

        gt_anchor_iou = bbox_overlaps(gt_boxes_wh, anchors)
        _, gt_anchor_argmax = gt_anchor_iou.max(1)

        for j, gt in enumerate(ground_truth[i]):
            gi = min(width - 1, max(0, int(gt_boxes[j, 0])))
            gj = min(height - 1, max(0, int(gt_boxes[j, 1])))
            best_anchor = gt_anchor_argmax[j]

            coord_target[i][best_anchor][0][gj * width + gi] = gt_boxes[j, 0] - gi
            coord_target[i][best_anchor][1][gj * width + gi] = gt_boxes[j, 1] - gj
            coord_target[i][best_anchor][2][gj * width + gi] = torch.log(
                max(gt_boxes[j, 2], 1.0) / anchors[best_anchor, 2])
            coord_target[i][best_anchor][3][gj * width + gi] = torch.log(
                max(gt_boxes[j, 3], 1.0) / anchors[best_anchor, 3])

            conf_target[i][best_anchor][gj * width + gi] = gt_pred_iou[j][
                best_anchor * height * width + gj * width + gi]
            cls_target[i][best_anchor][gj * width + gi] = int(gt[4])

            coord_mask[i][best_anchor][0][gj * width + gi] = 1
            cls_mask[i][best_anchor][gj * width + gi] = 1
            conf_mask[i][best_anchor][gj * width + gi] = object_scale

    coord_mask = coord_mask.expand_as(coord_target)
    cls_target = cls_target[cls_mask].view(-1).long()
    cls_mask = cls_mask.view(-1, 1).repeat(1, class_num)
    cls = cls[cls_mask].view(-1, class_num)

    if torch.cuda.is_available():
        coord_target = coord_target.cuda()
        conf_target = conf_target.cuda()
        cls_target = cls_target.cuda()
        coord_mask = coord_mask.cuda()
        conf_mask = conf_mask.cuda()
        cls_mask = cls_mask.cuda()

    conf_mask = conf_mask.sqrt()

    # compute loss
    MSE = nn.MSELoss(size_average=False)
    CrossEntropy = nn.CrossEntropyLoss(size_average=False)
    coord_loss = coord_scale * MSE(coord * coord_mask, coord_target * coord_mask) / batch_size
    conf_loss = MSE(conf * conf_mask, conf_target * conf_mask) / batch_size
    cls_loss = class_scale * 2 * CrossEntropy(cls, cls_target) / batch_size

    return coord_loss, conf_loss, cls_loss
















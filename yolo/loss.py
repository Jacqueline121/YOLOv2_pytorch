import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg
from utils.bbox_operation import bbox_overlaps, xyxy2xywh, xywh2xyxy


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

    coord = output[0]  # shape: (B, H*W, A, 4)
    conf = output[1]  # shape: (B, H*W, A, 1)
    cls = output[2]  # shape: (B*H*W*A, 20)

    gt_boxes_batch = ground_truth[0]
    gt_classes_batch = ground_truth[1]
    num_obj_batch = ground_truth[2]

    batch_size = conf.size(0)

    # calculate the coordinates of the predicted boxes

    # generate all anchors
    pred_boxes_batch = torch.FloatTensor(batch_size, height * width * anchor_num, 4)

    ctr_x = torch.range(0, width - 1).repeat(height, 1).contiguous().view(width * height)
    ctr_y = torch.range(0, height - 1).repeat(width, 1).t().contiguous().view(width * height)
    anchor_w = anchors[:, 2].contiguous().view(anchor_num, 1)
    anchor_h = anchors[:, 3].contiguous().view(anchor_num, 1)

    if torch.cuda.is_available():
        pred_boxes_batch = pred_boxes_batch.cuda()
        ctr_x = ctr_x.cuda()
        ctr_y = ctr_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()
        anchors = anchors.cuda()

    # bbox_transform_inv
    coord_copy = coord.clone()
    coord_copy = coord_copy.permute(0, 2, 3, 1)  # shape: (10, 5, 4, 169)
    pred_boxes_batch[:, :, 0] = (coord_copy[:, :, 0].detach() + ctr_x).view(batch_size, anchor_num*height*width)
    pred_boxes_batch[:, :, 1] = (coord_copy[:, :, 1].detach() + ctr_y).view(batch_size, anchor_num*height*width)
    pred_boxes_batch[:, :, 2] = (coord_copy[:, :, 2].detach() * anchor_w).view(batch_size, anchor_num*height*width)
    pred_boxes_batch[:, :, 3] = (coord_copy[:, :, 3].detach() * anchor_h).view(batch_size, anchor_num*height*width)

    #pred_boxes_batch = pred_boxes_batch.cpu()

    # build target
    # coord_target = torch.zeros(batch_size, anchor_num, 4, height * width, requires_grad=False)
    # conf_target = torch.zeros(batch_size, anchor_num, height * width, requires_grad=False)
    # cls_target = torch.zeros(batch_size, anchor_num, height * width, requires_grad=False)
    #
    # coord_mask = torch.zeros(batch_size, anchor_num, 1, height*width, requires_grad=False)
    # conf_mask = torch.ones(batch_size, anchor_num, height*width, requires_grad=False) * noobject_scale
    # cls_mask = torch.zeros(batch_size, anchor_num, height*width, requires_grad=False).byte()

    coord_target = torch.zeros(batch_size, height * width, anchor_num, 4, requires_grad=False)
    conf_target = torch.zeros(batch_size,  height * width, anchor_num, requires_grad=False)
    cls_target = torch.zeros(batch_size, height * width, anchor_num, requires_grad=False)

    coord_mask = torch.zeros(batch_size, height * width, anchor_num, 1, requires_grad=False)
    conf_mask = torch.ones(batch_size, height * width, anchor_num, requires_grad=False) * noobject_scale
    cls_mask = torch.zeros(batch_size, height * width, anchor_num, requires_grad=False).byte()

    for i in range(batch_size):
        # if len(gt_boxes_xyxy) == 0:
        #     continue

        num_obj = num_obj_batch[i]
        pred_boxes_xywh = pred_boxes_batch[i]
        gt_boxes_xyxy = gt_boxes_batch[i][:num_obj, :]  # shape: (num_obj, 4)
        gt_classes = gt_classes_batch[i][:num_obj]

       # gt_boxes_xywh = torch.zeros_like(gt_boxes_xyxy)  # shape: (num_obj, 4)
       # for j, gt in enumerate(gt_boxes_xyxy):
        gt_boxes_xywh = xyxy2xywh(gt_boxes_xyxy) / reduction

        pred_boxes_xyxy = xywh2xyxy(pred_boxes_xywh)
        gt_pred_iou = bbox_overlaps(gt_boxes_xyxy, pred_boxes_xyxy)  # shape: (num_obj, H*W*A)
        mask = (gt_pred_iou > thresh).sum(0) >= 1
        conf_mask[i][mask.view_as(conf_mask[i])] = 0  # object: 0, no-object: 1

        gt_boxes_wh = gt_boxes_xywh.clone()
        gt_boxes_wh[:, :2] = 0

        gt_boxes_wh = xywh2xyxy(gt_boxes_wh)
        anchors_xyxy = xywh2xyxy(anchors)
        gt_anchor_iou = bbox_overlaps(gt_boxes_wh, anchors_xyxy)  # shape: (num_obj, A)
        _, gt_anchor_argmax = gt_anchor_iou.max(1)   # shape: (num_obj, 1)?
        # the index of the best anchor for each gt_boxes

        for j, gt in enumerate(gt_boxes_xywh):
            gi = min(width - 1, max(0, int(gt_boxes_xywh[j, 0])))
            gj = min(height - 1, max(0, int(gt_boxes_xywh[j, 1])))
            best_anchor = gt_anchor_argmax[j]

            # coord_target[i][best_anchor][0][gj * width + gi] = gt_boxes_xywh[j, 0] - gi
            # coord_target[i][best_anchor][1][gj * width + gi] = gt_boxes_xywh[j, 1] - gj
            # coord_target[i][best_anchor][2][gj * width + gi] = torch.log(max(gt_boxes_xywh[j, 2], 1.0) / anchors[best_anchor, 2])
            # coord_target[i][best_anchor][3][gj * width + gi] = torch.log(max(gt_boxes_xywh[j, 3], 1.0) / anchors[best_anchor, 3])

            coord_target[i][gj * width + gi][best_anchor][0] = gt_boxes_xywh[j, 0] - gi
            coord_target[i][gj * width + gi][best_anchor][1] = gt_boxes_xywh[j, 1] - gj
            coord_target[i][gj * width + gi][best_anchor][2] = torch.log(
                max(gt_boxes_xywh[j, 2], 1.0) / anchors[best_anchor, 2])
            coord_target[i][gj * width + gi][best_anchor][3] = torch.log(
                max(gt_boxes_xywh[j, 3], 1.0) / anchors[best_anchor, 3])

            # conf_target[i][best_anchor][gj * width + gi] = gt_pred_iou[j][best_anchor * height * width + gj * width + gi]
            # cls_target[i][best_anchor][gj * width + gi] = int(gt[4])
            #
            # coord_mask[i][best_anchor][0][gj * width + gi] = 1
            # cls_mask[i][best_anchor][gj * width + gi] = 1
            # conf_mask[i][best_anchor][gj * width + gi] = object_scale

            conf_target[i][gj * width + gi][best_anchor] = gt_pred_iou[j][
                best_anchor * height * width + gj * width + gi]
            cls_target[i][gj * width + gi][best_anchor] = int(gt_classes[j])

            coord_mask[i][gj * width + gi][best_anchor][0] = 1  # object: 1, no-object: 0
            cls_mask[i][gj * width + gi][best_anchor] = 1   # object: 1, no-object: 0
            conf_mask[i][gj * width + gi][best_anchor] = object_scale
            # best bbox: object_scale, thresh>0.6 but not the best: 0, no-object: 1

    coord_mask = coord_mask.expand_as(coord_target)
    cls_target = cls_target[cls_mask].view(-1).long() - 1
    cls_mask = cls_mask.view(-1, 1).repeat(1, class_num)  # shape: (B*H*W*A, 20)
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
    # MSE = nn.MSELoss(size_average=False)
    # CrossEntropy = nn.CrossEntropyLoss(size_average=False)
    coord_loss = coord_scale * F.mse_loss(coord * coord_mask, coord_target * coord_mask, reduction='sum') / batch_size
    conf_loss = F.mse_loss(conf * conf_mask, conf_target * conf_mask, reduction='sum') / batch_size
    cls_loss = class_scale * 2 * F.cross_entropy(cls, cls_target, reduction='sum') / batch_size

    return coord_loss, conf_loss, cls_loss
















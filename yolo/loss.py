import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils.anchors import generate_anchors
from config.config import cfg
from utils.box_operation import box_transform, box_transform_inv, box_overlaps, xyxy2xywh, xywh2xyxy


def yolo_loss(output_pred, ground_truth, height, width):
    '''

    :param output_pred: is Variable
    :param ground_truth:  is data
    :param height:
    :param width:
    :return:
    '''

    coord_pred = output_pred[0].data  # (16, 196*5, 4)  data
    conf_pred = output_pred[1].data  # (16, 196*5, 1)
    cls_pred = output_pred[2].data  # (16*196*5, 20)

    gt_boxes = ground_truth[0]  # (16, 6, 4)， 6 is the num_obj
    gt_classes = ground_truth[1]  # (16, 6) data
    num_obj = ground_truth[2]  # (16， 1)

    batch_size = coord_pred.size(0)
    anchor_num = len(cfg.ANCHORS)

    cell_anchors_xywh = generate_anchors()  # (196*5, 4)

    anchors_xywh = cell_anchors_xywh.clone()
    anchors_xywh[:, 0:2] = anchors_xywh[:, 0:2] + 0.5

    if cfg.DEBUG:
        print('all cell:', cell_anchors_xywh[:12, :])
        print('all anchors:', anchors_xywh[:12, :])

    anchors_xyxy = xywh2xyxy(anchors_xywh)
    
    if torch.cuda.is_available():
        cell_anchors_xywh = cell_anchors_xywh.cuda()
        anchors_xyxy = anchors_xyxy.cuda()

    coord_target = coord_pred.new_zeros((batch_size, height * width, anchor_num, 4))
    conf_target = conf_pred.new_zeros((batch_size, height * width, anchor_num, 1))
    cls_target = cls_pred.new_zeros((batch_size, height * width, anchor_num, 1))

    coord_mask = coord_pred.new_zeros((batch_size, height * width, anchor_num, 1))
    conf_mask = conf_pred.new_ones((batch_size, height * width, anchor_num, 1)) * cfg.NO_OBJECT_SCALE
    cls_mask = cls_pred.new_zeros((batch_size, height * width, anchor_num, 1))

    for i in range(batch_size):
        gt_num = num_obj[i].item()
        gt_boxes_xyxy = gt_boxes[i, :gt_num, :]
        gt_class = gt_classes[i, :gt_num]

        gt_boxes_xyxy[:, 0::2] = gt_boxes_xyxy[:, 0::2] * width
        gt_boxes_xyxy[:, 1::2] = gt_boxes_xyxy[:, 1::2] * height

        gt_boxes_xywh = xyxy2xywh(gt_boxes_xyxy)

        # 1. calculate the predicted box
        pred_box_xywh = box_transform_inv(cell_anchors_xywh, coord_pred[i])
        pred_box_xyxy = xywh2xyxy(pred_box_xywh)

        # 2. calculate the IOU between each pred_box and gt_boxes
        pred_gt_iou = box_overlaps(pred_box_xyxy, gt_boxes_xyxy)  # conf_target (pred_num, gt_num)
        pred_gt_iou = pred_gt_iou.view(-1, anchor_num, gt_num)

        max_iou, _ = torch.max(pred_gt_iou, dim=-1, keepdim=True)

        if cfg.DEBUG:
            print('ious:', pred_gt_iou)

        num_pos = torch.nonzero(max_iou.view(-1) > cfg.THRESH).numel()
        if num_pos > 0:
            conf_mask[i][max_iou >= cfg.THRESH] = 0

        # 3. calculate the IOU between gt_boxes and anchors
        anchors_gt_iou = box_overlaps(anchors_xyxy, gt_boxes_xyxy).view(-1, anchor_num, gt_num)  # decide which anchor is responsible for the gt_box

        # 4. iterate over each gt_boxes
        for j in range(gt_num):
            gt_box_xywh = gt_boxes_xywh[j, :]
            g_x = torch.floor(gt_box_xywh[0])
            g_y = torch.floor(gt_box_xywh[1])
            # cell_idxth cell is responsible for this gt_box
            cell_idx = (g_y * width + g_x).long()

            best_anchor = torch.argmax(anchors_gt_iou[cell_idx, :, j])

            assigned_cell_anchor = cell_anchors_xywh.view(-1, anchor_num, 4)[cell_idx, best_anchor, :].unsqueeze(0)
            gt_box = gt_box_xywh.unsqueeze(0)
            target = box_transform(assigned_cell_anchor, gt_box)

            if cfg.DEBUG:
                print('assigned cell:', assigned_cell_anchor)
                print('gt:', gt_box)
                print('target:', target)

            coord_target[i, cell_idx, best_anchor, :] = target.unsqueeze(0)
            coord_mask[i, cell_idx, best_anchor, :] = 1

            conf_target[i, cell_idx, best_anchor, :] = max_iou[cell_idx, best_anchor, :]
            conf_mask[i, cell_idx, best_anchor, :] = cfg.OBJECT_SCALE

            if cfg.DEBUG:
                print('conf_target:', max_iou[cell_idx, best_anchor, :])

            cls_target[i, cell_idx, best_anchor, :] = gt_class[j]
            cls_mask[i, cell_idx, best_anchor, :] = 1

    coord_mask = coord_mask.expand_as(coord_target)

    coord_pred_variable, conf_pred_variable, cls_pred_variable = output_pred[0], output_pred[1], output_pred[2]

    coord_target = Variable(coord_target.view(batch_size, -1, 4))
    coord_mask = Variable(coord_mask.view(batch_size, -1, 4))
    conf_target = Variable(conf_target.view(batch_size, -1, 1))
    conf_mask = Variable(conf_mask.view(batch_size, -1, 1))
    cls_target = Variable(cls_target.view(-1).long())
    cls_mask = Variable(cls_mask.view(-1).long())

    keep = cls_mask.nonzero().squeeze(1)
    cls_pred_variable = cls_pred_variable[keep, :]
    cls_target = cls_target[keep] - 1

    # calculate loss
    coord_loss = cfg.COORD_SCALE * F.mse_loss(coord_pred_variable * coord_mask,
                                              coord_target * coord_mask, reduction='sum') / batch_size / 2.0
    conf_loss = F.mse_loss(conf_pred_variable * conf_mask,
                           conf_target * conf_mask, reduction='sum') / batch_size / 2.0
    cls_loss = cfg.CLASS_SCALE * F.cross_entropy(cls_pred_variable, cls_target, reduction='sum') / batch_size

    return coord_loss, conf_loss, cls_loss










import numpy as np
import torch
from config.config import cfg
from utils.anchors import generate_anchors
from utils.bbox_operation import xywh2xyxy, bbox_overlaps


def generate_pred_boxes(coord_pred):
    all_anchors = generate_anchors()

    pred_boxes = coord_pred.new(*coord_pred.size())

    pred_boxes[:, 0] = coord_pred[:, 0] + all_anchors[:, 0]
    pred_boxes[:, 1] = coord_pred[:, 1] + all_anchors[:, 1]
    pred_boxes[:, 2] = coord_pred[:, 2] + all_anchors[:, 2]
    pred_boxes[:, 3] = coord_pred[:, 3] + all_anchors[:, 3]

    return pred_boxes


def filter_boxes(pred_boxes, conf_pred, cls_pred, conf_thresh=0.6):
    cls_max, cls_argmax = torch.argmax(cls_pred, dim=-1, keepdim=True)
    cls_conf = conf_pred * cls_max
    keep = (cls_conf > conf_thresh).view(-1)

    keep_boxes = pred_boxes[keep, :]
    keep_conf = conf_pred[keep, :]
    keep_cls_max = cls_max[keep, :]
    keep_cls_argmax = cls_argmax[keep, :]

    return keep_boxes, keep_conf, keep_cls_max, keep_cls_argmax


def scale_boxes(boxes, im_info):
    h = im_info['height']
    w = im_info['width']

    input_h, input_w = cfg.TEST_SIZE
    scale_h, scale_w = input_h / h, input_w / w

    boxes *= cfg.REDUCTION

    boxes[:, 0::2] /= scale_w
    boxes[:, 1::2] /= scale_h

    boxes = xywh2xyxy(boxes)

    boxes[:, 0::2].clamp_(0, w-1)
    boxes[:, 1::2].clamp_(0, h-1)

    return boxes


def nms(boxes, conf, nms_thresh):
    conf_sort_index = torch.sort(conf, dim=0, descending=True)[1]
    keep = []
    while conf_sort_index.numel() > 0:
        i = conf_sort_index[0]
        keep.append(i)

        if conf_sort_index.numel == 1:
            break

        cur_box = boxes[conf_sort_index[0], :].view(-1, 4)
        res_box = boxes[conf_sort_index[1:], :].view(-1, 4)

        ious = bbox_overlaps(cur_box, res_box).view(-1)

        inds = torch.nonzero(ious < nms_thresh).squeeze()

        conf_sort_index = conf_sort_index[inds + 1].view(-1)

    return torch.LongTensor(keep)


def eval(output, im_info, conf_thresh, nms_thresh):

    coord_pred = output[0].view(-1, 4).cpu()  # shape: (A, H*W, 4)
    conf_pred = output[1].view(-1, 1).cpu()  # shape: (A, H*W, 1)
    cls_pred = output[2].view(-1, 20).cpu()  # shape: (A, H*W, 20)

    height = im_info['height']
    width = im_info['width']

    class_num = cls_pred.size(1)

    pred_boxes = generate_pred_boxes(coord_pred)

    boxes, conf, cls_max, cls_argmax = \
        filter_boxes(pred_boxes, conf_pred, cls_pred, conf_thresh)

    if boxes.size(0) == 0:
        return []

    boxes = scale_boxes(boxes, im_info)

    detections = []
    cls_argmax = cls_argmax.view(-1)

    # apply class-wise nms
    for cls in range(class_num):
        cls_mask = (cls_argmax == cls)
        inds = torch.nonzero(cls_mask).squeeze()

        if inds.numel() == 0:
            continue

        boxes_cls = boxes[inds, :].view(-1, 4)
        conf_cls = conf[inds, :].view(-1, 1)
        cls_max_cls = cls_max[inds].view(-1, 1)
        cls_label_cls = cls_argmax[inds].view(-1, 1)

        nms_keep = nms(boxes_cls, conf_cls.view(-2), nms_thresh)

        boxes_cls_keep = boxes_cls[nms_keep, :]
        conf_cls_keep = conf_cls[nms_keep, :]
        cls_max_cls_keep = cls_max_cls.view(-1, 1)[nms_keep, :]
        cls_label_cls_keep = cls_label_cls.view(-1, 1)[nms_keep, :]

        seq = [boxes_cls_keep, conf_cls_keep, cls_max_cls_keep, cls_label_cls_keep.float()]

        detections_cls = torch.cat(seq, dim=-1)
        detections.append(detections_cls)

    return torch.cat(detections, dim=0)








import torch
from config.config import cfg
from utils.anchors import generate_anchors
from utils.box_operation import box_transform_inv, xywh2xyxy, box_overlaps


def get_pred_boxes(coord):
    anchors = generate_anchors()
    pred_boxes = box_transform_inv(anchors, coord) # (x, y, w, h)
    return pred_boxes  # (H*W*A, 4)


def filter_boxes(pred_boxes, conf, cls, conf_thresh):
    cls_max, cls_argmax = torch.max(cls, dim=-1, keepdim=True)  # (H*W*A, 1)
    conf_cls = conf * cls_max

    keep = (conf_cls > conf_thresh).view(-1)

    pred_boxes_keep = pred_boxes[keep, :]
    conf_keep = conf[keep, :]
    cls_max_keep = cls_max[keep, :]
    cls_argmax_keep = cls_argmax[keep, :]

    return pred_boxes_keep, conf_keep, cls_max_keep, cls_argmax_keep


def rescale_boxes(pred_boxes, im_info):
    img_width = im_info[0].cpu().numpy()
    img_height = im_info[1].cpu().numpy()

    test_width, test_height = cfg.TEST_SIZE

    scale_width = test_width / img_width
    scale_height = test_height / img_height

    pred_boxes *= cfg.STRIDE

    pred_boxes[:, 0::2] /= scale_width
    pred_boxes[:, 1::2] /= scale_height

    pred_boxes = xywh2xyxy(pred_boxes)

    pred_boxes[:, 0::2].clamp_(0, img_width - 1)
    pred_boxes[:, 1::2].clamp_(0, img_height - 1)

    return pred_boxes


def nms(pred_boxes, conf, nms_thresh):
    order = torch.sort(conf, dim=0, descending=True)[1]
    keep = []

    while order.numel() > 0:
        keep.append(order[0])

        if order.numel() == 1:
            break

        cur_box = pred_boxes[order[0], :].view(-1, 4)
        res_boxes = pred_boxes[order[1:], :].view(-1, 4)

        ious = box_overlaps(cur_box, res_boxes).view(-1)

        idxs = torch.nonzero(ious < nms_thresh).squeeze()
        order = order[idxs + 1].view(-1)

    return torch.LongTensor(keep)


def eval(output, im_info, conf_thresh, nms_thresh):
    coord = output[0].view(-1, 4).cpu()   # (H*W*anchor_num, 4)
    conf = output[1].view(-1, 1).cpu()    # (H*W*anchor_num, 1)
    cls = output[2].view(-1, 20).cpu()    # (H*W*anchor_num, 20)

    # 1. generate predicted boxes
    pred_boxes = get_pred_boxes(coord)

    # 2. filter boxes whose conf is less than conf_thresh
    pred_boxes, conf, cls_max, cls_argmax = \
        filter_boxes(pred_boxes, conf, cls, conf_thresh)

    if pred_boxes.size(0) == 0:
        return []

    # 3. rescale the pred_boxes
    pred_boxes = rescale_boxes(pred_boxes, im_info)

    # 4. nms
    detections = []
    cls_argmax = cls_argmax.view(-1)
    for i in range(cfg.CLASS_NUM):
        idxs = torch.nonzero(cls_argmax == i).squeeze()

        if idxs.numel() == 0:
            continue

        pred_boxes_cls = pred_boxes[idxs, :].view(-1, 4)
        conf_cls = conf[idxs, :].view(-1, 1)
        cls_max_cls = cls_max[idxs, :].view(-1, 1)
        cls_argmax_cls = cls_argmax[idxs].view(-1, 1)

        nms_keep = nms(pred_boxes_cls, conf_cls, nms_thresh)

        pred_boxes_cls = pred_boxes_cls[nms_keep, :]
        conf_cls = conf_cls[nms_keep, :]
        cls_max_cls = cls_max_cls[nms_keep, :]
        cls_label = cls_argmax_cls[nms_keep, :].float()

        detections_cls = torch.cat([pred_boxes_cls, conf_cls, cls_max_cls, cls_label], dim=-1)

        detections.append(detections_cls)

    return torch.cat(detections, dim=0)

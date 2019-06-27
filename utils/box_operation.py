import torch


def box_overlaps2(boxes, query_boxes):

    N = boxes.size(0)
    K = query_boxes.size(0)

    boxes_area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).view(N, 1)
    query_boxes_area = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1]).view(1, K)

    ix1 = torch.max(boxes[:, 0].view(N, 1), query_boxes[:, 0].view(1, K))
    iy1 = torch.max(boxes[:, 1].view(N, 1), query_boxes[:, 1].view(1, K))
    ix2 = torch.min(boxes[:, 2].view(N, 1), query_boxes[:, 2].view(1, K))
    iy2 = torch.min(boxes[:, 3].view(N, 1), query_boxes[:, 3].view(1, K))

    iw = torch.max(ix2 - ix1 + 1, boxes.new(1).fill_(0))
    ih = torch.max(iy2 - iy1 + 1, boxes.new(1).fill_(0))

    inter_area = iw * ih
    union_area = boxes_area + query_boxes_area - inter_area

    IoU = inter_area / union_area

    return IoU


def box_overlaps(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """

    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious


def box_transform(box1, box2):
    t_ctr_x = box2[:, 0] - box1[:, 0]
    t_ctr_y = box2[:, 1] - box1[:, 1]
    t_w = box2[:, 2] / box1[:, 2]
    t_h = box2[:, 3] / box1[:, 3]

    t_ctr_x = t_ctr_x.view(-1, 1)
    t_ctr_y = t_ctr_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    target = torch.cat([t_ctr_x, t_ctr_y, t_w, t_h], dim=1)
    return target


def box_transform_inv(boxes, delta):

    ctr_x = boxes[:, 0] + delta[:, 0]
    ctr_y = boxes[:, 1] + delta[:, 1]
    w = boxes[:, 2] * delta[:, 2]
    h = boxes[:, 3] * delta[:, 3]

    ctr_x = ctr_x.view(-1, 1)
    ctr_y = ctr_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    pred_boxes = torch.cat([ctr_x, ctr_y, w, h], dim=1)

    return pred_boxes


def xyxy2xywh(box):
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]
    ctr_x = box[:, 0] + w / 2
    ctr_y = box[:, 1] + h / 2

    ctr_x = ctr_x.view(-1, 1)
    ctr_y = ctr_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([ctr_x, ctr_y, w, h], dim=1)
    return xywh_box


def xywh2xyxy(box):
    x1 = box[:, 0] - (box[:, 2]) / 2
    y1 = box[:, 1] - (box[:, 3]) / 2
    x2 = box[:, 0] + (box[:, 2]) / 2
    y2 = box[:, 1] + (box[:, 3]) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box







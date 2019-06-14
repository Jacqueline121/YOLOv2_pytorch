import torch
import torch.nn as nn
import torch.nn.functional as F
from .Darknet19 import Darknet19
from .Darknet19 import conv3x3, conv1x1
from .loss import yolo_loss
from config.config import cfg
import os


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


class YOLO(nn.Module):
    def __init__(self, pretrained=False):
        super(YOLO, self).__init__()

        anchors = cfg.ANCHORS
        self.class_num = cfg.CLASS_NUM
        self.anchor_num = len(anchors)
        self.anchors = anchors

        darknet = Darknet19(1000)

        if pretrained:
            weights_path = os.path.join('yolo', 'pretrained', 'darknet19_448.weights')
            darknet.load_weights(weights_path)

        self.layer1 = nn.Sequential(darknet.block1, darknet.block2, darknet.block3,
                                  darknet.block4, darknet.block5)

        # detection
        self.passthrough_layer = conv1x1(512, 64)

        self.layer2 = darknet.block6

        self.layer3 = conv3x3(1024, 1024)

        self.layer4 = conv3x3(1024, 1024)

        self.layer5 = conv3x3((1024+256), 1024)

        self.layer6 = nn.Conv2d(1024, self.anchor_num * (5 + self.class_num), 1, 1, 0, bias=False)

        self.reorg_layer = ReorgLayer()

    def forward(self, x, ground_truth=None):
        x = self.layer1(x)

        passthrough = self.passthrough_layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        passthrough = self.reorg_layer(passthrough)
        x = torch.cat([x, passthrough], dim=1)

        x = self.layer5(x)
        output = self.layer6(x)

        batch_size, _, height, width = output.data.size()

        output = output.view(batch_size, self.anchor_num, -1, height * width).permute(0, 3, 1, 2)

       # output = output.permute(0, 2, 3, 1).contiguous().view(batch_size, height*width, self.anchor_num, -1)
        # get predicted result
        coord = torch.zeros_like(output[:, :, :, :4])
        coord[:, :, :, :2] = output[:, :, :, :2].sigmoid()
        coord[:, :, :, 2:4] = output[:, :, :, 2:4].exp()
        conf = output[:, :, :, 4].sigmoid()
        cls = output[:, :, :, 5:].contiguous().view(-1, self.class_num)
        cls_pred = F.softmax(output[:, :, :, 5:], dim=-1)
        output = (coord, conf, cls)

        if self.training:
            coord_loss, conf_loss, cls_loss = yolo_loss(output, ground_truth, height, width)
            return coord_loss, conf_loss, cls_loss

        return coord, conf, cls_pred









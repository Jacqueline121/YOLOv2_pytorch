import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import Darknet19, conv1x1, conv3x3
from config.config import cfg
from .loss import yolo_loss


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


class YOLOv2(nn.Module):
    def __init__(self, pretrained=None):
        super(YOLOv2, self).__init__()

        anchors = cfg.ANCHORS
        self.anchor_num = len(anchors)
        self.class_num = 20

        darknet19 = Darknet19()

        if pretrained:
            print('load pretrained darknet19 model...')
            pretrained_path = os.path.join('yolo', 'pretrained', 'darknet19_448.weights')
            darknet19.load_weights(pretrained_path)

        self.layer1 = nn.Sequential(darknet19.block1, darknet19.block2, darknet19.block3,
                                    darknet19.block4, darknet19.block5)

        self.layer2 = darknet19.block6

        self.layer3 = conv3x3(1024, 1024)
        self.layer4 = conv3x3(1024, 1024)

        self.passthrough_layer = conv1x1(512, 64)

        self.layer5 = conv3x3((1024+256), 1024)

        self.layer6 = nn.Conv2d(1024, self.anchor_num*(5 + self.class_num), 1, 1, 0)

        self.reorg_layer = ReorgLayer()

    def forward(self, im_data, ground_truth=None):
        x = self.layer1(im_data)
        pass_through = self.passthrough_layer(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pass_through = self.reorg_layer(pass_through)

        x = torch.cat([pass_through, x], dim=1)
        x = self.layer5(x)
        output = self.layer6(x)

        if cfg.DEBUG:
            print('check output', output.view(-1)[0:10])

        batch_size, _, height, width = output.size()

        # operate output
        output = output.permute(0, 2, 3, 1).contiguous().view(batch_size, height*width*self.anchor_num, -1)

        coord = torch.zeros_like(output[:, :, 0:4])
        coord[:, :, 0:2] = output[:, :, 0:2].sigmoid()
        coord[:, :, 2:4] = output[:, :, 2:4].exp()
        conf = output[:, :, 4:5].sigmoid()
        cls = output[:, :, 5:].view(-1, self.class_num)
        cls_pred = F.softmax(output[:, :, 5:], dim=-1)

        if self.training:
            output_pred = (coord, conf, cls)
            coord_loss, conf_loss, cls_loss = yolo_loss(output_pred, ground_truth, height, width)

            return coord_loss, conf_loss, cls_loss

        return coord, conf, cls_pred


import torch.nn as nn
import torch.nn.functional as F
from utils.network import WeightLoader


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(out_channel),
                         nn.LeakyReLU(0.1, inplace=True))


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
                         nn.BatchNorm2d(out_channel),
                         nn.LeakyReLU(0.1, inplace=True))


def maxpool2d():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def softmax(dim=1):
    return nn.Softmax(dim=dim)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)

        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class Darknet19(nn.Module):
    def __init__(self, class_num):
        super(Darknet19, self).__init__()

        self.block1 = nn.Sequential(conv3x3(3, 32))

        self.block2 = nn.Sequential(maxpool2d(),
                                    conv3x3(32, 64))

        self.block3 = nn.Sequential(maxpool2d(),
                                    conv3x3(64, 128),
                                    conv1x1(128, 64),
                                    conv3x3(64, 128))

        self.block4 = nn.Sequential(maxpool2d(),
                                    conv3x3(128, 256),
                                    conv1x1(256, 128),
                                    conv3x3(128, 256))

        self.block5 = nn.Sequential(maxpool2d(),
                                    conv3x3(256, 512),
                                    conv1x1(512, 256),
                                    conv3x3(256, 512),
                                    conv1x1(512, 256),
                                    conv3x3(256, 512)
                                    )

        self.block6 = nn.Sequential(maxpool2d(),
                                    conv3x3(512, 1024),
                                    conv1x1(1024, 512),
                                    conv3x3(512, 1024),
                                    conv1x1(1024, 512),
                                    conv3x3(512, 1024)
                                    )

        self.conv = nn.Conv2d(1024, class_num, kernel_size=1, stride=1)
        self.avgpool = GlobalAvgPool2d()
        self.softmax = softmax()

    def forward(self, input):

        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.conv(out)
        out = self.avgpool(out)
        out = self.softmax(out)

        return out

    def load_weights(self, weight_path):
        weight_loader = WeightLoader()
        weight_loader.load(self, weight_path)

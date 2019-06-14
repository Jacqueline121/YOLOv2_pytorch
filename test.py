from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, detection_collate
from torch.optim import SGD
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from yolo.yolo import YOLO
from config.config import cfg
import time


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv2')
    parser.add_argument('--dataset', dest='--dataset', default='voc07test', type=str)
    parser.add_argument('--batch_size', dest='--batch_size', default=10, type=int)
    parser.add_argument('--cuda', dest='use_cuda', default=True, type=bool)
    parser.add_argument('--mGPUs', dest='mGPUs', default=True, type=bool)
    parser.add_argument('--num_workers', dest='num_workers', default=2, type=int)
    parser.add_argument('--vis', dest='vis', default=False, type=bool)
    parser.add_argument('--output_dir', dest='output_dir', default='output', type=str)
    parser.add_argument('--check_epoch', dest='check_epoch', default=160, type=int)

    args = parser.parse_args()

    return args


def test():
    args = parse_args()

    class_num = cfg.CLASS_NUM

    print('load data')
    if args.dataset == 'voc07test':
        dataset_name = 'voc_2007_test'
    elif args.dataset == 'voc12test':
        dataset_name = 'voc_2012_test'
    else:
        raise NotImplementedError

    test_dataset = RoiDataset(get_imdb(dataset_name))
    test_dataloder = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('load model')
    model = YOLO()
    pretrain_model = os.path.join(args.output_dir, 'yolov2_epoch_{}.pth'.format(args.check_epoch))

    if torch.cuda.is_available():
        state_dict = torch.load(pretrain_model)
    else:
        state_dict = torch.load(pretrain_model, map_location='cpu')

    model.load_state_dict(state_dict)

    if args.use_cuda:
        model.cuda()
    model.eval()

    dataset_size = len(test_dataset)

    all_boxes = [[[] for _ in range(dataset_size)] for _ in range(class_num)]

    det_file = os.path.join(args.output_dir, 'detections.pkl')

    img_id = -1

    with torch.no_grad():
        for batch, (im_data, im_infos) in enumerate(test_dataloder):
            if args.use_cuda:
                im_data = im_data.cuda()

            im_data_variable = Variable(im_data)

            outputs = model(im_data_variable)

            for i in range(im_data.size(0)):
                img_id += 1
                output = [item[i].data for item in outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = eval(output, im_info, )








from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, detection_collate
from torch.autograd import Variable
from yolo.yolov2 import YOLOv2
from config.config import cfg
from eval import eval
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from utils.network import WeightLoader
from utils.visualize import draw_detection_boxes
from utils.network import WeightLoader


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv2')
    parser.add_argument('--dataset', dest='dataset', default='voc07test', type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=2, type=int)
    parser.add_argument('--cuda', dest='use_cuda', default=True, type=bool)
    parser.add_argument('--mGPUs', dest='mGPUs', default=True, type=bool)
    parser.add_argument('--num_workers', dest='num_workers', default=2, type=int)
    parser.add_argument('--vis', dest='vis', default=False, type=bool)
    parser.add_argument('--output_dir', dest='output_dir', default='output', type=str)
    parser.add_argument('--check_epoch', dest='check_epoch', default=159, type=int)
    parser.add_argument('--conf_thresh', dest='conf_thresh', default=0.005, type=float)
    parser.add_argument('--nms_thresh', dest='nms_thresh', default=0.45, type=float)

    args = parser.parse_args()

    return args


def test():
    args = parse_args()

    if args.vis:
        args.conf_thresh = 0.5

    # load test data
    if args.dataset == 'voc07test':
        dataset_name = 'voc_2007_test'
    elif args.dataset == 'voc12test':
        dataset_name = 'voc_2012_test'
    else:
        raise NotImplementedError

    test_imdb = get_imdb(dataset_name)
    test_dataset = RoiDataset(test_imdb, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # load model
    model = YOLOv2()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    weight_file_path = os.path.join(args.output_dir, 'yolov2_epoch_{}.pth'.format(args.check_epoch))
    
    if torch.cuda.is_available:
        state_dict = torch.load(weight_file_path)
    else:
        state_dict = torch.load(weight_file_path, map_location='cpu')
    
    model.load_state_dict(state_dict['model'])

    if args.use_cuda:
        model = model.cuda()

    model.eval()

    num_data = len(test_dataset)

    all_boxes = [[[] for _ in range(num_data)] for _ in range(test_imdb.num_classes)]

    img_id = -1

    det_file = os.path.join(args.output_dir, 'detections.pkl')

    with torch.no_grad():
        for batch_size, (im_data, im_infos) in enumerate(test_dataloader):

            if args.use_cuda:
                im_data = im_data.cuda()
                im_infos = im_infos.cuda()

            im_data_variable = Variable(im_data)

            outputs = model(im_data_variable)

            for i in range(im_data.size(0)):
                img_id += 1

                output = [item[i].data for item in outputs]
                im_info = im_infos[i]

                detections = eval(output, im_info, args.conf_thresh, args.nms_thresh)

                if len(detections) > 0:
                    for i in range(cfg.CLASS_NUM):
                        idxs = torch.nonzero(detections[:, -1] == i).view(-1)
                        if idxs.numel() > 0:
                            cls_det = torch.zeros((idxs.numel(), 5))
                            cls_det[:, :4] = detections[idxs, :4]
                            cls_det[:, 4] = detections[idxs, 4] * detections[idxs, 5]
                            all_boxes[i][img_id] = cls_det.cpu().numpy()

                if args.vis:
                    img = Image.open(test_imdb.image_path_at(img_id))
                    if len(detections) == 0:
                        continue
                    det_boxes = detections[:, :5].cpu().numpy()
                    det_classes = detections[:, -1].long().cpu().numpy()

                    imshow = draw_detection_boxes(img, det_boxes, det_classes, class_names=test_imdb.classes)

                    plt.figure()
                    plt.imshow(imshow)
                    plt.show()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    test_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)


if __name__ == '__main__':
    test()
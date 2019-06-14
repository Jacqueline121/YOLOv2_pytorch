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
import time


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv2')
    parser.add_argument('--dataset', dest='dataset', default='voc0712trainval', type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', default=True, type=bool)
    parser.add_argument('--mGPUs', dest='mGPUs', default=True, type=bool)
    parser.add_argument('--pretrained', dest='pretrained', default=True, type=bool)
    parser.add_argument('--batch_size', dest='batch_size', default=10, type=int)
    parser.add_argument('--epochs', dest='epochs', default=160, type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=2, type=int)
    parser.add_argument('--lr', dest='lr', default=0.0001, type=float)
    parser.add_argument('--decay_lrs', dest='decay_lrs', default=[59, 89])
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.0005, type=float)
    parser.add_argument('--bais_decay', dest='bais_decay', default=False, type=bool)
    parser.add_argument('--gamma', dest='gamma', default=0.1, type=float)
    parser.add_argument('--use_tfboard', dest='use_tfboard', default=True, type=bool)
    parser.add_argument('--display_interval', dest='display_interval', default=100, type=int)
    parser.add_argument('--output_dir', dest='output_dir', default='output', type=str)
    parser.add_argument('--save_interval', dest='save_interval', default=20, type=int)
    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch', default=160, type=int)
    parser.add_argument('--resume', dest='resume', default=False, type=bool)

    args = parser.parse_args()
    return args


def get_dataset(dataset_names):
    names = dataset_names.split('+')
    dataset = RoiDataset(get_imdb(names[0]))
    print('load dataset {}'.format(names[0]))
    for name in names[1:]:
        temp = RoiDataset(get_imdb(name))
        dataset += temp
        print('load dataset {}'.format(name))
    return dataset


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():

    args = parse_args()
    lr = args.lr
    decay_lrs = args.decay_lrs
    momentum = args.momentum
    weight_decay = args.weight_decay
    gamma = args.gamma

    pretrained_model = os.path.join('data', 'pretrained', 'darknet19_448.weights')

    if args.use_tfboard:
        writer = SummaryWriter()

    # load data
    print('load data')
    if args.dataset == 'voc07trainval':
        dataset_name = 'voc_2007_trainval'
    elif args.dataset == 'voc12trainval':
        dataset_name = 'voc_2012_trainval'
    elif args.dataset == 'voc0712trainval':
        dataset_name = 'voc_2007_trainval+voc_2012_trainval'
    else:
        raise NotImplementedError

    train_dataset = get_dataset(dataset_name)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=detection_collate,
                                  drop_last=True)

    print('load model')
    model = YOLO(args.pretrained)

    print('optimizer')
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if args.use_gpu:
        model = model.cuda()

    if args.mGPUs:
        model = nn.DataParallel(model)

    model.train()

    iter_per_epoch = int(len(train_dataset) / args.batch_size)

    for epoch in range(args.epochs):
        loss_temp = 0
        tic = time.time()

        train_data_iter = iter(train_dataloader)

        if epoch in decay_lrs:
            lr = lr * gamma
            adjust_lr(optimizer, lr)

        # if multi scale
        # NotImplemented

        for step in range(iter_per_epoch):

            # if multi scale
            # NotImplemented

            im_data, gt_boxes, gt_classes, num_obj = next(train_data_iter)

            if args.use_gpu:
                im_data = im_data.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_classes = gt_classes.cuda()
                num_obj = num_obj.cuda()

            im_data_variable = Variable(im_data)
            # gt_classes = gt_classes.view(gt_classes.size(0), gt_classes.size(1), 1)
            # ground_truth = torch.cat([gt_boxes, gt_classes], dim=2)

            ground_truth = (gt_boxes, gt_classes, num_obj)

            coord_loss, conf_loss, cls_loss = model(im_data_variable, ground_truth)

            loss = coord_loss.mean() + conf_loss.mean() + cls_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

            if (step + 1) % args.display_interval == 0:
                toc = time.time()
                loss_temp /= args.display_interval

                coord_loss_m = coord_loss.mean().item()
                conf_loss_m = conf_loss.mean().item()
                cls_loss_m = cls_loss.mean().item()

                print("[epoch %2d][step %4d/%4d] loss: %.4f, lr: %.2e, time cost %.1fs" \
                      % (epoch, step + 1, iter_per_epoch, loss_temp, lr, toc - tic))
                print("\t\t\tcoord_loss: %.4f, conf_loss: %.4f, cls_loss: %.4f" \
                      % (coord_loss_m, conf_loss_m, cls_loss_m))

                if args.use_tfboard:
                    n_iter = epoch * iter_per_epoch + step + 1
                    writer.add_scalar('losses/loss', loss_temp, n_iter)
                    writer.add_scalar('losses/coord_loss', coord_loss_m, n_iter)
                    writer.add_scalar('losses/conf_loss', conf_loss_m, n_iter)
                    writer.add_scalar('losses/cls_loss', cls_loss_m, n_iter)

                loss_temp = 0
                tic = time.time()

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if epoch % args.save_interval == 0:
            save_name = os.path.join(args.output_dir, 'yolov2_epoch_{}.pth'.format(epoch))
            torch.save({
                'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                'epoch': epoch,
                'lr': lr
            }, save_name)


if __name__ == '__main__':
    train()
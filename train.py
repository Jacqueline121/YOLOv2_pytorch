import os
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
from dataset.roidb import RoiDataset, detection_collate
from dataset.factory import get_imdb
from yolo.yolov2 import YOLOv2
from tensorboardX import SummaryWriter


def get_dataset(datasetnames):
    names = datasetnames.split('+')
    dataset = RoiDataset(get_imdb(names[0]))
    print('load dataset {}'.format(names[0]))
    for name in names[1:]:
        tmp = RoiDataset(get_imdb(name))
        dataset += tmp
        print('load and add dataset {}'.format(name))
    return dataset


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv2')
    parser.add_argument('--dataset', dest='dataset', default='voc07trainval', type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', default=True, type=bool)
    parser.add_argument('--mGPUs', dest='mGPUs', default=True, type=bool)
    parser.add_argument('--pretrained', dest='pretrained', default=True, type=bool)
    parser.add_argument('--batch_size', dest='batch_size', default=16, type=int)
    parser.add_argument('--epochs', dest='epochs', default=160, type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=8, type=int)
    parser.add_argument('--lr', dest='lr', default=0.0001, type=float)
    parser.add_argument('--decay_lrs', dest='decay_lrs', default=[59, 89])
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.0005, type=float)
    parser.add_argument('--gamma', dest='gamma', default=0.1, type=float)
    parser.add_argument('--use_tfboard', dest='use_tfboard', default=True, type=bool)
    parser.add_argument('--display_interval', dest='display_interval', default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir', default='output', type=str)
    parser.add_argument('--save_interval', dest='save_interval', default=20, type=int)
    parser.add_argument('--resume', dest='resume', default=False, type=bool)

    args = parser.parse_args()
    return args


def train():
    args = parse_args()

    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    decay_lrs = args.decay_lrs
    gamma = args.gamma

    if args.use_tfboard:
        writer = SummaryWriter()

    # load data
    print('loading data')
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

    # model
    print('load model')
    model = YOLOv2(pretrained=args.pretrained)

    # optimizer
    optimizer = SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if args.use_gpu:
        model = model.cuda()

    if args.mGPUs:
        model = nn.DataParallel(model)

    model.train()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)

    for epoch in range(args.epochs):
        tic = time.time()
        loss_temp = 0

        train_data_iter = iter(train_dataloader)

        if epoch in decay_lrs:
            lr = lr * gamma
            print('change lr to {}'.format(lr))
            adjust_lr(optimizer, lr)

        for step in range(iters_per_epoch):
            im_data, gt_boxes, gt_classes, num_obj = next(train_data_iter)

            if args.use_gpu:
                im_data = im_data.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_classes = gt_classes.cuda()
                num_obj = num_obj.cuda()

            im_data_variabel = Variable(im_data)
            ground_truth = (gt_boxes, gt_classes, num_obj)

            coord_loss, conf_loss, cls_loss = model(im_data_variabel, ground_truth)

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
                      % (epoch, step + 1, iters_per_epoch, loss_temp, lr, toc - tic))
                print("\t\t\tcoord_loss: %.4f, conf_loss: %.4f, cls_loss: %.4f" \
                      % (coord_loss_m, conf_loss_m, cls_loss_m))

                if args.use_tfboard:
                    n_iter = epoch * iters_per_epoch + step + 1
                    writer.add_scalar('losses/loss', loss_temp, n_iter)
                    writer.add_scalar('losses/coord_loss', coord_loss_m, n_iter)
                    writer.add_scalar('losses/conf_loss', conf_loss_m, n_iter)
                    writer.add_scalar('losses/cls_loss', cls_loss_m, n_iter)

                loss_temp = 0
                tic = time.time()

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if (epoch + 1) % args.save_interval == 0:
            save_name = os.path.join(args.output_dir, 'yolov2_epoch_{}.pth'.format(epoch))
            torch.save({
                'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                'epoch': epoch,
                'lr': lr
            }, save_name)


if __name__ == '__main__':
    train()
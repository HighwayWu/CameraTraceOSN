import os
import cv2
import shutil
import random
import argparse
import numpy as np
import logging as logger

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
from sklearn.metrics import accuracy_score

from model import STATE as extractor


logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, data_size=512, crop_rep=4, val_ratio=None):
        self.data_root = data_root
        self.data_size = data_size
        self.crop_rep = crop_rep
        self.train_list = []
        self.id2image_path_list = {}
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path, image_label = line.split(' ')
            label = int(image_label)
            self.train_list.append((image_path, label))

            if not label in self.id2image_path_list.keys():
                self.id2image_path_list[label] = []
            self.id2image_path_list[label].append(image_path)
            line = train_file_buf.readline().strip()

        self.isVal = False
        if val_ratio is not None:
            np.random.shuffle(self.train_list)
            self.test_list = self.train_list[-int(len(self.train_list) * val_ratio):]
            self.train_list = self.train_list[:-int(len(self.train_list) * val_ratio)]

        self.aug_scenario = A.Compose([
            A.RandomScale(scale_limit=(-0.4, 0.0), p=1.0),
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.OneOf([
                A.ImageCompression(quality_lower=75, quality_upper=95, compression_type=0, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
            ], p=0.8),
        ])

        self.albu_pre = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.RandomRotate90(p=0.33),
            A.Flip(p=0.33),
        ], p=1.0)
        self.imagenet_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.osn_list = ['nat', 'natFBH', 'natFBL', 'natWA']

    def transform(self, x):
        x = self.albu_pre(image=x)['image']
        x = self.imagenet_norm(x).unsqueeze(0)
        return x

    def __len__(self):
        if self.isVal:
            return len(self.test_list)
        else:
            return len(self.train_list)

    def __getitem__(self, index):
        if self.isVal:
            return self.getitem(index, self.test_list)
        else:
            return self.getitem(index, self.train_list)

    def getitem(self, index, data_list):
        image_path, image_label = data_list[index]

        image = cv2.imread(os.path.join(self.data_root, image_path))
        crop_list, label_list = None, []
        for _ in range(self.crop_rep):
            crop = self.transform(image)
            crop_list = torch.cat([crop_list, crop]) if crop_list is not None else crop
            label_list.append(image_label)
        label_list = torch.LongTensor(label_list)
        return crop_list, label_list

    def set_val_True(self):
        self.isVal = True

    def set_val_False(self):
        self.isVal = False


def train_one_epoch(data_loader, model, optimizer, cur_epoch, loss_meter, args):
    loss_meter.reset()
    n_fc = args.n_fc
    split_list = args.split_list
    batch_idx = 0
    for (images, labels) in data_loader:
        b, rep, C, H, W = images.shape
        images = images.cuda().reshape(b * rep, C, H, W)
        labels = labels.cuda().flatten().squeeze()

        trace = model(images)

        idx_group = []
        for sid in range(1, len(split_list)):
            cur_group = [idx for idx, value in enumerate(labels) if split_list[sid-1] <= value < split_list[sid]]
            if len(cur_group) == 0:
                continue
            idx_group.append(cur_group)

        # Skip if the current batch does not contain enough scenarios
        if len(idx_group) != n_fc:
            continue

        loss_list = []
        for idx, group in enumerate(idx_group):
            cur_prob = model.module.fc_forward(model.module._prepare_rep(trace[group], idx), idx)
            loss_list.append(F.cross_entropy(cur_prob, torch.remainder(labels[group], 29)))

        optimizer.zero_grad()
        L = torch.stack(loss_list, dim=0)
        loss = torch.mean(L, dim=0)

        # Backward via Momentum Masking, rather than naive loss.backward()
        model.module.backward(L, idx_group)

        optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])
        if batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            logger.info('Ep %03d, it %03d/%03d, CE %7.6f' % (cur_epoch, batch_idx, len(data_loader), loss_avg))
            loss_meter.reset()
        if batch_idx > 500:
            break
        batch_idx += 1

    return loss_avg


def val_one_epoch(data_loader, model, args):
    data_loader.dataset.set_val_True()
    model.eval()
    split_list = args.split_list
    gt_labels_list, pred_labels_list = [], []
    for (images, labels) in data_loader:
        images = images.cuda()
        b, rep, C, H, W = images.shape
        images = images.reshape(b * rep, C, H, W)
        labels = labels.cuda().flatten().squeeze()

        idx_group = []
        for sid in range(1, len(split_list)):
            cur_group = [idx for idx, value in enumerate(labels) if split_list[sid-1] <= value < split_list[sid]]
            if len(cur_group) == 0:
                continue
            idx_group.append(cur_group)

        with torch.no_grad():
            trace = model(images)

            for idx, group in enumerate(idx_group):
                fc_prob = model.module.fc_forward(trace[group], idx)
                pred_label = torch.argmax(fc_prob, dim=1)

                gt_labels_list += list(torch.remainder(labels[group], 29).cpu().numpy())
                pred_labels_list += list(pred_label.cpu().numpy())

    score = accuracy_score(gt_labels_list, pred_labels_list)
    model.train()
    return score


def train(args):
    val_ratio = 0.1
    data_loader = DataLoader(
        ImageDataset(args.data_root, args.train_file, data_size=args.data_size,
                     crop_rep=args.crop_rep, val_ratio=val_ratio),
        args.batch_size, shuffle=True, num_workers=min(48, args.batch_size))

    model = extractor(args.num_class, args.n_fc)
    model = torch.nn.DataParallel(model).cuda()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.lr)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5)

    loss_meter = AverageMeter()
    val_best = -1
    for epoch in range(args.epoches):
        model.train()
        data_loader.dataset.set_val_False()
        train_one_epoch(data_loader, model, optimizer, epoch, loss_meter, args)

        isBest = ''
        val_score = val_one_epoch(data_loader, model, args)
        if val_score > val_best:
            val_best = val_score
            saved_name = 'Ep%03d_%5.4f.pt' % (epoch, val_score)
            isBest = '(Best)'
        else:
            saved_name = 'latest.pt'
        torch.save(model.state_dict(), os.path.join(args.out_dir, saved_name))
        logger.info('Score: Val Acc: %5.4f %s' % (val_score, isBest))
        lr_schedule.step(val_score)


if __name__ == '__main__':
    conf = argparse.ArgumentParser()
    conf.add_argument("--data_root", type=str, default='data/',
                      help="The root folder of training set.")
    conf.add_argument("--train_file", type=str,
                      default='data/Anno/Train_VISION_cls116.txt',
                      help="The training file path.")
    conf.add_argument("--num_class", type=int, default=29, help='The class number of training dataset')
    conf.add_argument("--n_fc", type=int, default=4, help='The number of scenarios')
    conf.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    conf.add_argument("--out_dir", type=str, default='out_dir', help="The folder to save models.")
    conf.add_argument('--epoches', type=int, default=9999, help='The training epoches.')
    conf.add_argument('--batch_size', type=int, default=16, help='The training batch size over all gpus.')
    conf.add_argument('--crop_rep', type=int, default=1, help='The crop time for each train image.')
    conf.add_argument('--data_size', type=int, default=224, help='The image size for training.')
    conf.add_argument('--gpu', type=str, default='7,6,5,4', help='The gpu')
    args = conf.parse_args()
    os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), os.cpu_count()))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    args.criterion_ce = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    args.split_list = list(range(0, 290, 29))[:args.n_fc + 1]

    logger.info(args)
    train(args)

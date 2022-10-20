import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torchvision import transforms
from model import STATE as extractor
from sklearn import metrics as skmetrics


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Preprocess:
    def __init__(self):
        self.data_size = 1536
        self.albu_pre = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.CenterCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)
        self.imagenet_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def process(self, image):
        image = self.albu_pre(image=image)['image']
        image = self.imagenet_norm(image)
        return image


if __name__ == '__main__':
    preprocess = Preprocess()
    weights_path = 'weights/EffNetB0_weights.pt'
    extractor = extractor()
    extractor.load_state_dict(torch.load(weights_path))
    extractor.eval()
    extractor.cuda()

    anno_path = 'data/Anno/'
    test_file = anno_path + 'Test_SIHDR_weibo.txt'
    csv_file = anno_path + 'Test_SIHDR_weibo_pair.csv'
    image_list = []

    image_list_buf = open(test_file)
    line = image_list_buf.readline().strip()
    while line:
        line_strs = line.split(' ')
        image_list.append(line_strs[0])
        line = image_list_buf.readline().strip()

    image_name2feature = {}
    with torch.no_grad():
        for image_name in image_list:
            x = cv2.imread(image_name)
            x = preprocess.process(x)
            x = torch.stack([x]).cuda()

            y = extractor(x)[0]
            y = y.detach().cpu().numpy()
            y = y / np.linalg.norm(y)
            image_name2feature[image_name] = y

    subsets_score_list, subsets_label_list = [], []

    csv_file = pd.read_csv(csv_file, sep=',', header=None).values[1:]
    for idx, item in enumerate(csv_file):
        imgA, imgB, label = item[0], item[1], int(item[2])
        if imgA in image_name2feature.keys() and imgB in image_name2feature.keys():
            feat1 = image_name2feature[imgA]
            feat2 = image_name2feature[imgB]
            cur_score = np.dot(feat1, feat2)
            subsets_score_list.append(cur_score)
            subsets_label_list.append(label)

    fpr, tpr, _ = skmetrics.roc_curve(subsets_label_list, subsets_score_list, pos_label=1)
    auc_value = skmetrics.auc(fpr, tpr)
    print('Open-set Verification (AUC):')
    print('%s: %5.4f' % (test_file, auc_value))

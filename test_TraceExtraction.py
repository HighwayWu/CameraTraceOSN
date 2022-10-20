import os
import cv2
import torch
import numpy as np
import albumentations as A
from torchvision import transforms
from model import STATE as extractor

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

    data_root = 'data/sample/'
    img1 = cv2.imread(data_root + 'A01_SDR_HAND_20.jpg')
    img2 = cv2.imread(data_root + 'A01_SDR_HAND_8.jpg')
    img3 = cv2.imread(data_root + 'A02_SDR_HAND_006.jpg')

    feat = []
    for x in [img1, img2, img3]:
        x = preprocess.process(x)
        x = torch.stack([x]).cuda()

        y = extractor(x)[0]
        y = y.detach().cpu().numpy()
        y = y / np.linalg.norm(y)
        feat.append(y)

    cos_sim1 = np.dot(feat[0], feat[1])  # same camera, cos_sim should be higher
    cos_sim2 = np.dot(feat[0], feat[2])  # diff camera, cos_sim should be lower
    print('Cosine Similarity:')
    print('SAME Camera Model: %.4f (Expected: 0.6760)' % cos_sim1)
    print('DIFF Camera Model: %.4f (Expected: 0.1139)' % cos_sim2)

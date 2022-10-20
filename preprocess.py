import os
import cv2
import random
import numpy as np
import pandas as pd


data_root = 'data/'
save_path = 'data/Anno/'
osn_list = [
    # 'facepost',
    # 'instagram',
    # 'telegram',
    # 'twitter',
    # 'whatsapp',
    # 'ding',
    # 'qq',
    # 'wechat',
    'weibo',
]


def TrainFile_Generation_VISION():
    osn_list = [
        '_nat',
        '_natFBH',
        '_natFBL',
        '_natWA',
    ]
    dup_list = [
        'D10_Apple_iPhone4s',
        'D14_Apple_iPhone5c',
        'D15_Apple_iPhone6',
        'D18_Apple_iPhone5c',
        'D26_Samsung_GalaxyS3Mini',
        'D34_Apple_iPhone5',
    ]
    dup_list = [item[:3] for item in dup_list]
    path = 'data/VISION/VISION'
    flist = sorted(os.listdir(path + osn_list[0]))
    img_list = []
    name2label = []
    for osn_label, osn in enumerate(osn_list):
        for file in flist:
            if file[:3] in dup_list:
                continue
            path2 = path + osn + '/' + file + '/'
            vision_name = osn + '_' + file
            if vision_name not in name2label:
                name2label.append(vision_name)
            label = name2label.index(vision_name)
            flist2 = sorted(os.listdir(path2))
            for file2 in flist2:
                if ' ' in file2:
                    continue
                img_list.append((path2 + file2, label))

    print('#Images of VISION: %d' % len(img_list))
    print('#Classes of VISION: %d' % len(name2label))
    textfile = open(save_path + 'Train_VISION_cls%d.txt' % len(name2label), 'w')
    for item in img_list:
        textfile.write('%s %s\n' % (item[0], item[1]))
    textfile.close()


def TestFile_Generation_FODB():
    dup_list = [
        'D16_Samsung_GalaxyA6_2',
        'D25_Huawei_P9lite_2',
    ]
    dup_list = [item[:3] for item in dup_list]
    path = data_root + 'FODB/'
    flist = sorted(os.listdir(path))
    img_list = []
    for file in flist:
        if file[0] != 'D':
            continue
        if file[:3] in dup_list:
            continue
        path2 = path + file + '/orig/'
        flist2 = sorted(os.listdir(path2))
        np.random.shuffle(flist2)
        tmp_cnt = 0
        for file2 in flist2:
            if '.jpg' not in file2:
                continue
            img_list.append((path2 + file2, int(file2[1:3]) - 1))
            tmp_cnt += 1
            if tmp_cnt >= 25:
                break

    img_list = sorted(img_list)
    print('#Images of FODB: %d' % len(img_list))
    textfile = open(save_path + 'Test_FODB_orig.txt', 'w')
    for item in img_list:
        textfile.write('%s %s\n' % (item[0], item[1]))
    textfile.close()

    gt_name_dict = {}
    gt_name_list = []
    for filename, gt_label in img_list:
        if gt_label in gt_name_dict.keys():
            name_list = gt_name_dict[gt_label]
            name_list.append(filename)
        else:
            name_list = [filename]
        gt_name_dict.update({gt_label: name_list})
        gt_name_list.append((gt_label, filename))

    images_A, images_B, predictions = [], [], []
    pos_all_list = []
    for label in gt_name_dict.keys():
        idx_list = gt_name_dict[label]
        list_len = len(idx_list)
        for i in range(list_len-1):
            for j in range(i+1, list_len):
                pos_all_list.append((idx_list[i], idx_list[j]))
    np.random.shuffle(pos_all_list)
    for na, nb in pos_all_list[:5000]:
        images_A.append(na)
        images_B.append(nb)
        predictions.append(1)
    pos_num = len(images_A)
    num_neg_per_label = len(images_A) // len(gt_name_list) + 1

    most_hard = []
    for idx, (label, filename) in enumerate(gt_name_list):
        for _ in range(num_neg_per_label):
            rnd_idx = np.random.randint(0, len(gt_name_list))
            if gt_name_list[rnd_idx][0] == gt_name_list[idx][0]:
                continue
            most_hard.append((gt_name_list[idx][1], gt_name_list[rnd_idx][1]))
    np.random.shuffle(most_hard)
    for na, nb in most_hard[:pos_num]:
        images_A.append(na)
        images_B.append(nb)
        predictions.append(0)

    df = pd.DataFrame({
        'imageA': images_A,
        'imageB': images_B,
        'label': predictions,
    })
    df.to_csv(save_path + 'Test_FODB_orig_pair.csv', index=False, header=True)
    # print('Total images num %d' % len(img_list))
    # print('Total pairs num %d (pos %d, neg %d)' % (len(predictions), pos_num, len(predictions) - pos_num))

    for osn in osn_list:
        textfile = open(save_path + 'Test_FODB_%s.txt' % osn, 'w')
        for item in img_list:
            textfile.write('%s %s\n' % (item[0].replace('orig', osn), item[1]))
        textfile.close()
        images_A_tmp = [item.replace('orig', osn) for item in images_A]
        images_B_tmp = [item.replace('orig', osn) for item in images_B]
        df = pd.DataFrame({
            'imageA': images_A_tmp,
            'imageB': images_B_tmp,
            'label': predictions,
        })
        df.to_csv(save_path + 'Test_FODB_%s_pair.csv' % osn, index=False, header=True)


def TestFile_Generation_SIHDR():
    path = data_root + 'SIHDR/orig/'
    flist = sorted(os.listdir(path))
    np.random.shuffle(flist)
    img_list = []
    label_list = []
    label_cnt = np.zeros(25)
    for file in flist:
        anno = file[:3]
        if anno not in label_list:
            label_list.append(anno)
        label = label_list.index(anno)
        if label_cnt[label] >= 25:
            continue
        else:
            label_cnt[label] += 1
        img_list.append((path + file, label))

    img_list = sorted(img_list)
    print('#Images of SIHDR: %d' % len(img_list))
    textfile = open(save_path + 'Test_SIHDR_orig.txt', 'w')
    for item in img_list:
        textfile.write('%s %s\n' % (item[0], item[1]))
    textfile.close()

    gt_name_dict = {}
    gt_name_list = []
    for filename, gt_label in img_list:
        if gt_label in gt_name_dict.keys():
            name_list = gt_name_dict[gt_label]
            name_list.append(filename)
        else:
            name_list = [filename]
        gt_name_dict.update({gt_label: name_list})
        gt_name_list.append((gt_label, filename))

    images_A, images_B, predictions = [], [], []
    pos_all_list = []
    for label in gt_name_dict.keys():
        idx_list = gt_name_dict[label]
        list_len = len(idx_list)
        for i in range(list_len-1):
            for j in range(i+1, list_len):
                pos_all_list.append((idx_list[i], idx_list[j]))
    np.random.shuffle(pos_all_list)
    for na, nb in pos_all_list[:5000]:
        images_A.append(na)
        images_B.append(nb)
        predictions.append(1)
    pos_num = len(images_A)
    num_neg_per_label = len(images_A) // len(gt_name_list) + 1

    most_hard = []
    for idx, (label, filename) in enumerate(gt_name_list):
        for _ in range(num_neg_per_label):
            rnd_idx = np.random.randint(0, len(gt_name_list))
            if gt_name_list[rnd_idx][0] == gt_name_list[idx][0]:
                continue
            most_hard.append((gt_name_list[idx][1], gt_name_list[rnd_idx][1]))
    np.random.shuffle(most_hard)
    for na, nb in most_hard[:pos_num]:
        images_A.append(na)
        images_B.append(nb)
        predictions.append(0)

    df = pd.DataFrame({
        'imageA': images_A,
        'imageB': images_B,
        'label': predictions,
    })
    df.to_csv(save_path + 'Test_SIHDR_orig_pair.csv', index=False, header=True)
    # print('Total images num %d' % len(img_list))
    # print('Total pairs num %d (pos %d, neg %d)' % (len(predictions), pos_num, len(predictions) - pos_num))

    for osn in osn_list:
        textfile = open(save_path + 'Test_SIHDR_%s.txt' % (osn), 'w')
        for item in img_list:
            textfile.write('%s %s\n' % (item[0].replace('orig', osn), item[1]))
        textfile.close()
        images_A_tmp = [item.replace('orig', osn) for item in images_A]
        images_B_tmp = [item.replace('orig', osn) for item in images_B]
        df = pd.DataFrame({
            'imageA': images_A_tmp,
            'imageB': images_B_tmp,
            'label': predictions,
        })
        df.to_csv(save_path + 'Test_SIHDR_%s_pair.csv' % osn, index=False, header=True)


if __name__ == '__main__':
    # generate file list for training
    # E.g., Train_VISION.txt contains [[image_path_1, image_label_1], [image_path_2, image_label_2], ...]
    TrainFile_Generation_VISION()

    # generate file list for Open-set Verification
    # E.g., Test_FODB.txt contains [[image_path_1, image_label_1], [image_path_2, image_label_2], ...]
    # E.g., Test_FODB_pair.csv contains [[image_path_1, image_path_2, isSameOrNot], ...]
    TestFile_Generation_FODB()
    TestFile_Generation_SIHDR()

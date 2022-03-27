#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random

import torch.utils.data
from loguru import logger

import cv2
import numpy as np
import json
import glob
import hashlib

#from ..dataloading import get_yolox_datadir
#from .datasets_wrapper import Dataset
from yolox.data.datasets.datasets_wrapper import Dataset
from DOTA_devkit import dota_utils
from pathlib import Path
from tqdm import tqdm

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

dotav10_classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

dotav15_classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
                'container-crane']

class DOTADataset(Dataset):
    def __init__(self, name="train", data_dir=None, img_size=(1024, 1024), preproc=None, cache=False, save_result_dir=None):
        super().__init__(img_size)
        self.imgs = None
        self.name = name
        self.data_dir = data_dir
        self.img_size = img_size
        self.labels_dir = os.path.join(data_dir, name, 'labelTxt')
        self.imgs_dir = os.path.join(data_dir, name, 'images')
        self.preproc = preproc
        self.save_result_dir = save_result_dir
        self.labels_file = [files for root, dirs, files in os.walk(self.labels_dir)]
        self.labels_file = [os.path.join(self.labels_dir, file) for file in self.labels_file[0]]
        self.imgs_file   = [file.replace('labelTxt', 'images').replace('.txt', '.png') for file in self.labels_file]
        assert len(self.labels_file) ==  len(self.imgs_file)
        self.imgs_num = len(self.imgs_file)
        self.class_id = {}
        for i, cls in enumerate(dotav15_classes):
            self.class_id[cls] = i

        self.ids = [i for i in range(len(self.labels_file))]
        random.shuffle(self.ids)

        if cache :
            self._cache_images()

    def __len__(self):
        return self.imgs_num

    def load_image(self, index):
        print(self.imgs_file[index])
        return cv2.imread(self.imgs_file[index])

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def pull_item(self, index):
        img = self.load_image(index)
        height, width = img.shape[0], img.shape[1]
        ann_file = self.labels_file[index]
        objects = dota_utils.parse_dota_poly2(ann_file)
        targets = []
        for obj in objects:
            class_id = self.class_id[obj['name']]
            poly     = obj['poly']
            targets.append([class_id] + poly)

        return None, None,None,None

    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target, img_info, img_id = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

    def letterbox(self, img, new_shape=(1024,1024), color=(114,114,114),auto=False, scaleFill=False, scale_up=True, stride=32):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] /  shape[1])
        if not scale_up:
            r = min(r, 1.0)
        radio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dh, dw = np.mod(dh, stride), np.mod(dw, stride)
        elif scaleFill:
            dh, dw = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            radio = new_shape[1]/shape[1], new_shape[0]/shape[0]

        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, radio, (dw, dh)

    def mixup(self, im, labels, im2, labels2):
        r = np.random.beta(32.0, 32.0)
        im = (im * r +im2 * (1-r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        return im, labels







if __name__ == '__main__':
    dataset = DOTADataset(name='train', data_dir='/home/yanggang/diskPoints/work2/DOTA_SPLIT')
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    img = np.zeros([800, 1224,3])
    im, r, pad = dataset.letterbox(img)
    cv2.imshow("img",im)
    cv2.waitKey()
    print(im, r, pad)
    print("ending")
    #for i, (img, target, img_info, img_id) in enumerate(dataloader):
    #    print("reading one batch")
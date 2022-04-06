#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import cv2
import numpy as np

from yolox.utils import adjust_box_anns, get_local_rank
from yolox.data.data_augment import random_affine,random_cutout
from yolox.data.datasets.datasets_wrapper import Dataset
from dota import DOTADataset,collate_fn,draw,drawOneImg
#from ..data_augment import random_affine
#from .datasets_wrapper import Dataset


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicOrientedDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, random_affine_enable=False,
        random_cutout_enable=True, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.random_affine_enable = random_affine_enable
        self.random_cutout_enable = random_cutout_enable
        self.local_rank = get_local_rank()

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 1] = scale * _labels[:, 1] + padw
                    labels[:, 2] = scale * _labels[:, 2] + padh
                    labels[:, 3] = scale * _labels[:, 3] + padw
                    labels[:, 4] = scale * _labels[:, 4] + padh
                    labels[:, 5] = scale * _labels[:, 5] + padw
                    labels[:, 6] = scale * _labels[:, 6] + padh
                    labels[:, 7] = scale * _labels[:, 7] + padw
                    labels[:, 8] = scale * _labels[:, 8] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)

            if self.random_affine_enable:
                mosaic_img, mosaic_labels = random_affine(
                    mosaic_img,
                    mosaic_labels,
                    target_size=(2 * input_w, 2 * input_h),
                    degrees=self.degrees,
                    translate=self.translate,
                    scales=self.scale,
                    shear=self.shear,
                    oriented=True
                )


            if self.random_cutout_enable:
                mosaic_img, mosaic_labels, _ = random_cutout(
                    mosaic_img,
                    mosaic_labels,
                    xc=xc,
                    yc=yc,
                    origin_size=(2 * input_w,  2 * input_h),
                    target_size=(input_w, input_h),
                    scope=0.0
                )

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return mix_img, padded_labels, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        r = np.random.beta(32.0, 32.0)
        HFLIP = random.uniform(0, 1) > 0.5
        VFLIP = random.uniform(0, 1) > 0.5
        cp_index = random.randint(0, self.__len__() - 1)
        cp_img, cp_labels, _, _ = self._dataset.pull_item(cp_index)
        if cp_img.shape != origin_img.shape:
            cp_img, scale, (padw, padh) = self._dataset.letterbox(cp_img)
            cp_labels[:, 1:-1:2] = scale[1] * cp_labels[:, 1:-1:2]
            cp_labels[:, 2:-1:2] = scale[0] * cp_labels[:, 2:-1:2]
            cp_labels[:, 1:-1:2] += int(padw)
            cp_labels[:, 2:-1:2] += int(padh)
            #drawOneImg(cp_img, cp_labels)
        width, height = cp_img.shape[0], cp_img.shape[1]

        if HFLIP: #horizontal flip
            cp_img = cp_img[:, ::-1, :].copy()
            cp_labels[:, 1:-1:2] = width - cp_labels[:, 1:-1:2]
        if VFLIP: #vertical flip
            cp_img = cp_img[::-1, :, :].copy()
            cp_labels[:, 2:-1:2] = height - cp_labels[:, 2:-1:2]

        img = (origin_img * r + (1 - r) * cp_img).astype(np.uint8)
        labels = np.concatenate((origin_labels, cp_labels), 0)
        #draw(img, labels, origin_img, origin_labels,cp_img, cp_labels)
        return img, labels


if __name__ == "__main__":

    from yolox.data.data_augment import OrientedValTransform, OrientedTrainTransform
    import torch
    dataset = DOTADataset(name='train', data_dir='/home/yanggang/data/DOTA_SPLIT')
    dataset = MosaicOrientedDetection(dataset,
                                      img_size=(1024, 1024),
                                      mosaic=True,
                                      preproc=OrientedTrainTransform())

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=False,collate_fn=collate_fn)

    for i, (img, label, img_info, img_id) in enumerate(dataloader):
        print("one batch")

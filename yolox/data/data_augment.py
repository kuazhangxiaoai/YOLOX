#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh, xyxy2cxcywhab
#from yolox.data.datasets.dota import draw

def draw(img, label, savepath=False, windowName='image'):
    for i, poly in enumerate(label):
        poly = np.float32(poly.reshape(4, 2))
        rect = cv2.minAreaRect(poly)
        label[i] = cv2.boxPoints(rect).reshape(-1)
    pts = label
    for i, poly in enumerate(pts):
        poly = poly.reshape([4, 2]).astype(np.int32)
        cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

    if savepath:
        cv2.imwrite(savepath, img)
    else:
        cv2.namedWindow(windowName,  0)
        cv2.imshow(windowName, img)
        cv2.waitKey()

def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
    # no return needed
    return cv2.cvtColor(img_hsv.astype(img.dtype), code=cv2.COLOR_HSV2BGR)


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale

def apply_affine_to_orientedbboxes(targets, target_size, M, scale):
    num_gts = len(targets)
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:,:2] = targets[:, 1:-1].reshape(4 * num_gts, 2) #x1y1, x2y2, x3y3,x4y4
    corner_points = corner_points @ M.T
    corner_points = corner_points.reshape(num_gts, 8)
    corner_points[:, 0::2] = corner_points[:, 0::2].clip(0, twidth)
    corner_points[:, 1::2] = corner_points[:, 1::2].clip(0, theight)
    targets[:, 1: -1] = corner_points
    return targets

def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets

def random_cutout(img, targets, xc, yc,origin_size=(2048, 2048), target_size=(1024, 1024), scope=0.3):
    assert origin_size[0] > target_size[0]
    assert origin_size[1] > target_size[1]
    x_min, y_min = max(0, xc - target_size[0]), max(0, yc - target_size[1])
    x_max, y_max = min(xc, target_size[0]), min(yc, target_size[1])
    x_min, y_min = int(x_min + scope * (x_max - x_min)), int(y_min + scope * (y_max - y_min))
    x_max, y_max = int(x_max - scope * (x_max - x_min)), int(y_max - scope * (y_max - y_min))
    start_x = random.randint(x_min, x_max)
    start_y = random.randint(y_min, y_max)
    img = img[start_y : start_y + target_size[1], start_x : start_x + target_size[0], :]
    if targets.size > 0:
        targets[:, 1] = targets[:, 1] - start_x
        targets[:, 2] = targets[:, 2] - start_y
        targets[:, 3] = targets[:, 3] - start_x
        targets[:, 4] = targets[:, 4] - start_y
        targets[:, 5] = targets[:, 5] - start_x
        targets[:, 6] = targets[:, 6] - start_y
        targets[:, 7] = targets[:, 7] - start_x
        targets[:, 8] = targets[:, 8] - start_y

        center_x = 0.25 * (targets[:, 1] + targets[:, 3] + targets[:, 5] + targets[:, 7])
        center_y = 0.25 * (targets[:, 2] + targets[:, 4] + targets[:, 6] + targets[:, 8])

        targets = targets[(center_x > 0) * (center_x < target_size[0]) * (center_y > 0) * (center_y < target_size[1])]

        #np.clip(targets[:, 1], 0, target_size[0], out=targets[:, 1])
        #np.clip(targets[:, 2], 0, target_size[1], out=targets[:, 2])
        #np.clip(targets[:, 3], 0, target_size[0], out=targets[:, 3])
        #np.clip(targets[:, 4], 0, target_size[1], out=targets[:, 4])
        #np.clip(targets[:, 5], 0, target_size[0], out=targets[:, 5])
        #np.clip(targets[:, 6], 0, target_size[1], out=targets[:, 6])
        #np.clip(targets[:, 7], 0, target_size[0], out=targets[:, 7])
        #np.clip(targets[:, 8], 0, target_size[1], out=targets[:, 8])

    return img, targets,(start_x, start_y)

def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    oriented=False
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) and (oriented is False) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)
    elif len(targets) and (oriented):
        targets = apply_affine_to_orientedbboxes(targets, target_size, M, scale)

    return img, targets


def _mirror(image, boxes, prob=0.5): #the format of boexes must be xyxy
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 0::2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels

class OrientedTrainTransform:
    def __init__(self, flip_prob=0.5, hsv_prob=1.0):
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, 1:-1].copy()
        labels = targets[:, -1].copy()
        image, boxes = _mirror(image, boxes)

        image_o = image.copy()
        height_o, width_o, _ = image_o.shape

        if random.random() < self.hsv_prob:
            image = augment_hsv(image)
        height, width, _ = image.shape

        # bbox_o: [xyxy] to [c_x, c_y, w, h, alpha, beta]
        #draw(image, boxes)
        boxes = xyxy2cxcywhab(boxes)

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((boxes_t, labels_t))

        return image, targets_t

class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))

class OrientedValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
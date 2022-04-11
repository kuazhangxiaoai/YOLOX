#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import time

import numpy as np

import torch
import torchvision
import cv2

from yolox.utils import nms_rotated

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "xyxy2cxcywhab"
]

def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def xyxy2cxcywhab(bboxes):
    for i, poly in enumerate(bboxes):
        poly = np.float32(poly.reshape(4,2))
        rect = cv2.minAreaRect(poly)
        bboxes[i] = cv2.boxPoints(rect).reshape(-1)
    boxes_num = bboxes.shape[0]
    new_boxes = np.zeros([boxes_num, 6],dtype=bboxes.dtype)
    xs, ys = bboxes[:, 0::2], bboxes[:, 1::2]
    x_min, x_max = xs.min(axis=1), xs.max(axis=1)
    y_min, y_max = ys.min(axis=1), ys.max(axis=1)
    new_boxes[:, 0], new_boxes[:, 1] = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5 # get cx and cy
    new_boxes[:, 2], new_boxes[:, 3] = (x_max - x_min), (y_max - y_min)             # get width and height
    x_min_index, x_max_index = np.argmin(xs, axis=1), np.argmax(xs, axis=1)
    y_min_index, y_max_index = np.argmin(ys, axis=1), np.argmax(ys, axis=1)
    new_boxes[:, 4] = xs[np.arange(boxes_num),y_min_index] - x_min                  #get alpha
    new_boxes[:, 5]  = y_max - ys[np.arange(boxes_num),x_max_index]                 # get beta
    return new_boxes

def xywhab2xyxy(bboxes, device='cuda'):
    #bboxes (num, [cx, cy, w, h, alpha, beta]): described by format of xywhab
    #return (num, [x1, y1, x2, y2, x3, y3, x4, y4]): described by format of xyxy
    boxes_num = bboxes.shape[0]
    new_boxes = torch.zeros([boxes_num, 8]).cuda()
    new_boxes[:, 0] = bboxes[:, 0] - 0.5 * bboxes[:, 2] + bboxes[:, 4]
    new_boxes[:, 1] = bboxes[:, 1] - 0.5 * bboxes[:, 3]
    new_boxes[:, 2] = bboxes[:, 0] + 0.5 * bboxes[:, 2]
    new_boxes[:, 3] = bboxes[:, 1] + 0.5 * bboxes[:, 3] - bboxes[:, 5]
    new_boxes[:, 4] = bboxes[:, 0] + 0.5 * bboxes[:, 2] - bboxes[:, 4]
    new_boxes[:, 5] = bboxes[:, 1] + 0.5 * bboxes[:, 3]
    new_boxes[:, 6] = bboxes[:, 0] - 0.5 * bboxes[:, 2]
    new_boxes[:, 7] = bboxes[:, 1] - 0.5 * bboxes[:, 3] + bboxes[:, 5]
    return new_boxes

def polyab_nms(dets, conf_thr=0.25, iou_thr=0.01, argnostic=False,classes=None, labels=(), multi_label=True, max_det=1500):
    #dets (batch_n, anchors_n, 7+num_classes) : [cx, cy, w, h, alpha, beta, conf, num_classes]
    nc = dets.shape[2] - 7
    xc = dets[..., 6] > conf_thr
    class_index = nc + 7
    max_wh = 4096
    max_nms = 30000
    time_limit = 30.0
    multi_label &= nc > 1

    t = time.time()
    output = [torch.zeros((0, 10), device=dets.device)] * predictions.shape[0]
    for xi, x in enumerate(dets):
        x = x[xc[xi]]
        x[:, 7:] *= x[:, 6:7]
        boxes = xywhab2xyxy(x[:, :6])

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 7), device=x.device)
            v[:, :6] = l[:, 1:7]
            v[:, 6] = 1.0
            v[range(len(l)), l[: 0].long() + 7] = 1.0
            x = torch.cat((x,v), 0)

        if not x.shape[0]:
            continue

        #Detections matrix n * 10 (x1, y1, x2, y2, x3, y3, x4, y4, conf, cls)
        if multi_label:
            i, j = (x[:, 7:class_index] > conf_thr).nonzero(as_tuple=False).T
            x = torch.cat((boxes[i], x[i, j+7, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 7:class_index].max(1, keepdim=True)
            x = torch.cat((boxes, conf, j.float()), 1)[conf.view(-1) > conf_thr]

        if classes is not None:
            x = x[(x[:, 9:10] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 8].argsort(descending=True)[:max_nms]]

        c = x[:, 9:10] * (0 if argnostic else max_wh)
        polys = x[:, :8].clone() + c
        scores = x[:, 8].unsqueeze(dim=1)
        dets = torch.cat([polys, scores], dim=1)
        _, i = nms_rotated.poly_nms(dets, iou_thr=iou_thr)
        if i.shape[0] > max_det:
            i = i[:, max_det]
        output[xi] = x[i]
        if time.time() - t > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


if __name__ == "__main__":
    polys = torch.Tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
         [0.1, 0.1, 0.1, 1.1, 1.1, 1.1, 1.1, 0.1]]
    ).numpy()
    predictions = torch.randn([4, 8500, 23]).cuda()
    polyab_nms(dets=predictions)
    print("ending")
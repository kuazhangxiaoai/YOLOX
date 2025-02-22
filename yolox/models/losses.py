#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 6)
        target = target.view(-1, 6)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:4] / 2), (target[:, :2] - target[:, 2:4] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:4] / 2), (target[:, :2] + target[:, 2:4] / 2)
        )

        area_p = torch.prod(pred[:, 2:4], 1)
        area_g = torch.prod(target[:, 2:4], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:4] / 2), (target[:, :2] - target[:, 2:4] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:4] / 2), (target[:, :2] + target[:, 2:4] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class OrientedIOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(OrientedIOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
        self.loss_mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 6)
        target = target.view(-1, 6)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:4] / 2), (target[:, :2] - target[:, 2:4] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:4] / 2), (target[:, :2] + target[:, 2:4] / 2)
        )

        area_p = torch.prod(pred[:, 2:4], 1)
        area_g = torch.prod(target[:, 2:4], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        loss_alpha = self.loss_mse(pred[:, 4], target[:, 4])
        loss_beta  = self.loss_mse(pred[:, 5], target[:, 5])

        if self.loss_type == "iou":
            loss = (1 - iou ** 2) + loss_alpha + loss_beta
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:4] / 2), (target[:, :2] - target[:, 2:4] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:4] / 2), (target[:, :2] + target[:, 2:4] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = (1 - giou.clamp(min=-1.0, max=1.0)) + loss_alpha + loss_beta

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

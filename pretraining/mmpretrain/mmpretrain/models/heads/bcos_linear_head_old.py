# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cls_head import ClsHead

# bcos imports
from bcos.modules import BcosLinear, BcosConv2d, LogitLayer
from bcos.modules import norms


@MODELS.register_module()
class BcosLinearClsHeadOld(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hid_channels: int = None,
                 num_layers: int = 1,
                 with_last_bn: bool = True,
                 with_last_bn_affine: bool = True,
                 init_cfg: Optional[dict] = None,
                 logit_bias: Optional[float] = None,
                 logit_temperature: Optional[float] = None,
                 with_avg_pool: bool = False,
                 **kwargs):
        super(BcosLinearClsHeadOld, self).__init__(init_cfg=init_cfg, **kwargs)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        if num_layers == 1:
            if hid_channels is not None:
                assert hid_channels == num_classes, "Since num_layers == 1, hid_channels should be equal to num_classes"
            elif hid_channels is None:
                hid_channels = num_classes

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        # for avg pool
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = BcosConv2d(in_channels, num_classes,kernel_size=1)
        else:
            self.fc = BcosLinear(in_channels, num_classes)
        """
        self.logit_layer = LogitLayer(
                logit_temperature=logit_temperature,
                logit_bias=logit_bias or -math.log(num_classes - 1),
                )
        """
        self.logit_layer = nn.Identity()

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        pre_logits = self.fc(pre_logits)
        cls_score = pre_logits
        
        # check for with_avg_pool flag
        if self.with_avg_pool:
            cls_score = self.avgpool(cls_score)

        # adding for Bcos
        cls_score = cls_score.flatten(1)
        cls_score = self.logit_layer(cls_score)
        return cls_score

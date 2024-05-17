# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS

# bcos dependencies
from bcos.modules.bcoslinear import *
from bcos.modules import norms


@MODELS.register_module()
class SwAVBcosNeck(BaseModule):
    """The non-linear neck of SwAV: fc-bn-relu-fc-normalization.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global average pooling after
            backbone. Defaults to True.
        with_l2norm (bool): whether to normalize the output after projection.
            Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        with_avg_pool: bool = True,
        with_l2norm: bool = True,
        norm_cfg: dict = None,
        init_cfg: Optional[Union[dict, List[dict]]] = None
    ) -> None:
        super().__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.with_l2norm = with_l2norm
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if out_channels == 0:
            self.projection_neck = nn.Identity()
        elif hid_channels == 0:
            self.projection_neck = BcosLinear(in_channels, out_channels)
        else:
            #self.norm = build_norm_layer(norm_cfg, hid_channels)[1]
            #self.norm = norms.NoBias(norms.DetachableLayerNorm)
            self.norm = norms.NoBias(norms.BatchNormUncentered1d)
            self.projection_neck = nn.Sequential(
                BcosLinear(in_channels, hid_channels),
                self.norm(hid_channels),
                #nn.ReLU(inplace=True),
                BcosLinear(hid_channels, out_channels),
            )

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Compute projection.

        Args:
            x (torch.Tensor): The feature vectors after pooling.

        Returns:
            torch.Tensor: The output features with projection or L2-norm.
        """
        x = self.projection_neck(x)
        if self.with_l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Forward function.

        Args:
            x (List[torch.Tensor]): list of feature maps, len(x) according to
                len(num_crops).

        Returns:
            torch.Tensor: The projection vectors.
        """
        avg_out = []
        for _x in x:
            _x = _x[0]
            if self.with_avg_pool:
                _out = self.avgpool(_x)
                avg_out.append(_out)
        feat_vec = torch.cat(avg_out)  # [sum(num_crops) * N, C]
        feat_vec = feat_vec.view(feat_vec.size(0), -1)
        output = self.forward_projection(feat_vec)
        return output

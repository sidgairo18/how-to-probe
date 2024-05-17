# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cls_head import ClsHead

# bcos imports
from bcos.modules import LogitLayer
from bcos.modules import norms


@MODELS.register_module()
class MultiLayerLinearClsHead(ClsHead):
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
        init_self.with_avg_pool = with_avg_poolcfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hid_channels: int,
                 num_layers: int = 1,
                 with_last_bn: bool = False,
                 with_last_bn_affine: bool = True,
                 with_bias: Optional[bool] = True,
                 logit_bias: Optional[float] = None,
                 logit_temperature: Optional[float] = None,
                 with_avg_pool: bool = False,
                 is_bcos: bool = False,
                 norm_cfg: dict = dict(type='BN1d'),
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(MultiLayerLinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

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
        self.relu = nn.ReLU(inplace=True)

        # for avg_pool
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                                   
            self.fc0 = nn.Conv2d(self.in_channels, self.hid_channels, kernel_size=1, bias=with_bias)
            if is_bcos:                                                                   
                self.bn0 = norms.NoBias(norms.BatchNormUncentered2d)(hid_channels)        
            else:                                                                         
                self.bn0 = nn.BatchNorm2d(self.hid_channels, affine=True)                 
            # for further layers                                                          
            self.fc_names = []                                                            
            self.bn_names = []                                                            
            for i in range(1, num_layers):                                                
                this_channels = num_classes if i == num_layers - 1 \
                        else hid_channels                                                 
                if i != num_layers - 1:                                                   
                    self.add_module(f'fc{i}',                                             
                            nn.Conv2d(hid_channels, this_channels, kernel_size=1, bias=with_bias))
                    if is_bcos:                                                           
                        self.add_module(f'bn{i}',                                         
                                        norms.NoBias(norms.BatchNormUncentered2d)(this_channels))
                    else:                                                                 
                        self.add_module(f'bn{i}',                                         
                                        nn.BatchNorm2d(this_channels, affine=True))       
                    self.bn_names.append(f'bn{i}')                                        
                else:                                                                     
                    self.add_module(f'fc{i}',                                             
                            nn.Conv2d(hid_channels, this_channels, kernel_size=1, bias=with_bias))
                    if with_last_bn:                                                      
                        if is_bcos:                                                       
                            self.add_module(f'bn{i}',                                     
                                            norms.NoBias(norms.BatchNormUncentered2d)(this_channels, affine=with_last_bn_affine))
                        else:                                                             
                            self.add_module(f'bn{i}',                                     
                                            nn.BatchNorm2d(this_channels, affine=with_last_bn_affine))
                        self.bn_names.append(f'bn{i}')                                    
                    else:                                                                 
                        self.bn_names.append(None)                                        
                self.fc_names.append(f'fc{i}')
        else:
            self.fc0 = nn.Linear(self.in_channels, self.hid_channels, bias=with_bias)     
            if is_bcos:                                                                   
                self.bn0 = norms.NoBias(norms.BatchNormUncentered1d)(hid_channels)        
            else:                                                                         
                self.bn0 = build_norm_layer(norm_cfg, hid_channels)[1]                    
            # for further layers                                                          
            self.fc_names = []                                                            
            self.bn_names = []                                                            
            for i in range(1, num_layers):                                                
                this_channels = num_classes if i == num_layers - 1 \
                        else hid_channels                                                 
                if i != num_layers - 1:                                                   
                    self.add_module(f'fc{i}',                                             
                            nn.Linear(hid_channels, this_channels, bias=with_bias))       
                    if is_bcos:                                                           
                        self.add_module(f'bn{i}',                                         
                                        norms.NoBias(norms.BatchNormUncentered1d)(this_channels))
                    else:                                                                 
                        self.add_module(f'bn{i}',                                         
                                        build_norm_layer(norm_cfg, this_channels)[1])     
                    self.bn_names.append(f'bn{i}')                                        
                else:                                                                     
                    self.add_module(f'fc{i}',                                             
                            nn.Conv2d(hid_channels, this_channels, kernel_size=1, bias=with_bias))
                    if with_last_bn:                                                      
                        if is_bcos:                                                       
                            self.add_module(f'bn{i}',                                     
                                            norms.NoBias(norms.BatchNormUncentered1d)(this_channels, affine=with_last_bn_affine))
                        else:                                                             
                            self.add_module(f'bn{i}',                                     
                                            build_norm_layer(                             
                                                dict(**norm_cfg, affine=with_last_bn_affine),
                                                this_channels)[1])                        
                        self.bn_names.append(f'bn{i}')                                    
                    else:                                                                 
                        self.bn_names.append(None)                                        
                self.fc_names.append(f'fc{i}')

        if num_layers == 1:                                                               
            self.bn0 = nn.Identity()                                                      
                                                                                          
        if logit_bias == 0 and logit_temperature is None:                                 
            self.logit_layer = nn.Identity()                                              
        else:                                                                             
            self.logit_layer = LogitLayer(                                                
                    logit_temperature=logit_temperature,                                  
                    logit_bias=logit_bias or -math.log(num_classes - 1),                  
                    )


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
        pre_logits = self.fc0(pre_logits)                                                 
        pre_logits = self.bn0(pre_logits)

        for fc_name, bn_name in zip(self.fc_names, self.bn_names):                        
            # apply relu                                                                  
            pre_logits = self.relu(pre_logits)                                            
            fc = getattr(self, fc_name)                                                   
            pre_logits = fc(pre_logits)                                                   
            if bn_name is not None:                                                       
                bn = getattr(self, bn_name)                                               
                pre_logits = bn(pre_logits)                                               
                                                                                          
        cls_score = pre_logits                                                            
                                                                                          
        # check for with_avg_pool flag                                                    
        if self.with_avg_pool:                                                            
            cls_score = self.avgpool(cls_score)                                           
                                                                                          
        # adding for the logit_layer                                                      
        cls_score = cls_score.flatten(1)                                                  
        cls_score = self.logit_layer(cls_score)

        return cls_score

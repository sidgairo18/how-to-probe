# Copyright (c) OpenMMLab. All rights reserved.
from .beitv2_neck import BEiTV2Neck
from .cae_neck import CAENeck
from .densecl_neck import DenseCLNeck
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .itpn_neck import iTPNPretrainDecoder
from .linear_neck import LinearNeck
from .mae_neck import ClsBatchNormNeck, MAEPretrainDecoder
from .milan_neck import MILANPretrainDecoder
from .mixmim_neck import MixMIMPretrainDecoder
from .mocov2_neck import MoCoV2Neck
from .nonlinear_neck import NonLinearNeck
from .simmim_neck import SimMIMLinearDecoder
from .spark_neck import SparKLightDecoder
from .swav_neck import SwAVNeck
from .dino_neck import DINONeck
# import bcos models here
from .mocov2_bcos_neck import MoCoV2BcosNeck
from .swav_bcos_neck import SwAVBcosNeck
from .bcosnonlinear_neck import BcosNonLinearNeck

__all__ = [
    'GlobalAveragePooling',
    'GeneralizedMeanPooling',
    'HRFuseScales',
    'LinearNeck',
    'BEiTV2Neck',
    'CAENeck',
    'DenseCLNeck',
    'MAEPretrainDecoder',
    'ClsBatchNormNeck',
    'MILANPretrainDecoder',
    'MixMIMPretrainDecoder',
    'MoCoV2Neck',
    'NonLinearNeck',
    'SimMIMLinearDecoder',
    'SwAVNeck',
    'iTPNPretrainDecoder',
    'SparKLightDecoder',
    'DINONeck',
    'MoCoV2BcosNeck',
    'SwAVBcosNeck',
    'BcosNonLinearNeck'
]

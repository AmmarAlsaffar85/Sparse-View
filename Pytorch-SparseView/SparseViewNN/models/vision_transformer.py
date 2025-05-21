# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, img_size: int, num_classes=1, zero_head=False, vis=False, patch_size=4,in_chans=1, embed_dim=96, depths=(2, 2, 6, 2),depths_decoder=[1, 2, 2, 2],
                       num_heads=[3, 6, 12, 24],window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                       drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                       norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                       use_checkpoint=False, final_upsample="expand_first"):
                 
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        
        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                patch_size=patch_size,
                                in_chans=in_chans,
                                num_classes=self.num_classes,
                                embed_dim=embed_dim,
                                depths=depths,
                                depths_decoder=depths_decoder,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                ape=ape,
                                patch_norm=patch_norm,
                                use_checkpoint=use_checkpoint)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,1,1,1)
        logits = self.swin_unet(x)
        return logits

    
    
   

"""
Model architectures. Modified from AVT
"""


import torch
import torch.nn as nn
from munch import DefaultMunch
from collections import OrderedDict
from torchvision.models import resnet101, resnet50

import yaml 
import os
from torchvision.models.video.resnet import (
    BasicBlock,
    Bottleneck,
    R2Plus1dStem,
    _video_resnet,
)
# from pretrainedmodels import bninception
import models.revit_model as revit_model
import timm
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
    resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked
from models.timm_viy import *
from types import MethodType
# from transformers import VivitConfig, VivitModel
# from transformers import XCLIPVisionModel, XCLIPVisionConfig
import math

def process_each_frame(model, video, *args, **kwargs):
    """
    Pass in each frame separately
    Args:
        video (B, C, T, H, W)
    Returns:
        feats: (B, C', T, 1, 1)
    """
    batch_size = video.size(0)
    time_dim = video.size(2)
    video_flat = video.transpose(1, 2).flatten(0, 1)
    feats_flat = model(video_flat, *args, **kwargs)
    return feats_flat.view((batch_size, time_dim) +
                           feats_flat.shape[1:]).transpose(
                               1, 2).unsqueeze(-1).unsqueeze(-1)


class FrameLevelModel(nn.Module):
    """Runs a frame level model on all the frames."""
    def __init__(self, num_classes: int, model: nn.Module = None):
        del num_classes
        super().__init__()
        self.model = model

    def forward(self, video, *args, **kwargs):
        return process_each_frame(self.model, video, *args, **kwargs)



def forward_(self, x):
    x = self.forward_features(x)
    return x


class TIMMModelTS(nn.Module):
    def __init__(self,
                 num_classes,
                 model_type='vit_base_patch16_224',
                 drop_cls=True):
        super(TIMMModelTS, self).__init__()
        model = timm.create_model(model_type,
                                  num_classes=0 if drop_cls else num_classes)
        self.model = model
        self.model.forward = MethodType(forward_, self.model)

    def forward(self, video, batch_size):
        # batch_size = video.size(0)
        # time_dim = video.size(2)
        video_flat = video.transpose(1, 2).flatten(0, 1)
        print(video_flat.shape)
        return self.model(video_flat)


class ResNet101(nn.Module):
    def __init__(self,
                num_classes,
                 model_type='vit_base_patch16_224',
                 drop_cls=True):
        super(ResNet101, self).__init__()
        model = resnet50()
        self.model = model
        self.model.fc = nn.Identity() 

    def forward(self, video, batch_size):
        video_flat = video.transpose(1, 2).flatten(0, 1)
        return self.model(video_flat)
    

class TIMMModelSwin(nn.Module):
    def __init__(self,
                 num_classes,
                 model_type='mvitv2_base',
                 drop_cls=True):
        super(TIMMModelSwin, self).__init__()
        model = timm.create_model(model_type,
                                  num_classes=0 if drop_cls else num_classes)
        self.model = model
        self.model.forward = MethodType(forward_, self.model)

    def forward(self, video, batch_size):
        batch_size = video.size(0)
        time_dim = video.size(2)
        video_flat = video.transpose(1, 2).flatten(0, 1)
        return self.model(video_flat)


class PositionalEncoding(nn.Module):
    """For now, just using simple pos encoding from language.
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



__MODELS__ = [
    "ReViT", "ReSwin", "ReMViTv2"
]

def build_ReViT(cfg):
    """
    Helper function to build neural backbone

    Input:
        cfg: configuration dictionary
    
    Returns:
        net: Neural network moduel, nn.Module object
    
    Raises:
        ValueError: Model is not supported
    """
    # print(cfg.MODEL)

    model_name = cfg.MODEL.name

    if model_name == "ReMViTv2" or model_name =="remvitv2":
        net = revit_model.ReViT(cfg)

    elif model_name == "ReViT" or model_name =="ReViT":
        net = revit_model.ReViT(cfg)

    elif "Hug" in model_name:
        net = timm.create_model("vit_base_patch16_224", pretrained=cfg.TRAIN.enable)
        net.head = revit_model.TransformerBasicHead(dim_in=768, num_classes=cfg.MODEL.num_classes)
    else:
        raise ValueError(f"Model name not supported, please inset one of the following: {__MODELS__}")
    print(f"Model {model_name} built successfully")   
    return net


def load_checkpoint(net, cfg, device):
    """
    Loads chekpoints into ReViT
    """
    mapping = "{rank}".format(rank=device)
    checkpoint = torch.load(cfg.SOLVER.load_path, map_location=mapping)
    model_state_dict = OrderedDict()

    try:
        for k, v in checkpoint['model'].items():
            model_state_dict[k.replace('backbone.', "")] = v
    except:
        model_state_dict = checkpoint['model_state_dict']

    net_dict = net.state_dict()
    print(net_dict.keys())
    print(model_state_dict.keys())
    pretrained_dict = {k: v for k, v in model_state_dict.items() if k in net_dict}
    missing_keys, unexp_keys = net.load_state_dict(pretrained_dict, strict=False)
    print(f"Unexpected keys: {unexp_keys}")
    print(f"Missing keys: {missing_keys}")
    print("Weights Loaded succesfully!!!")

    return 

class ReViT(nn.Module):
    def __init__(self,
                 num_classes,
                 model_type='',
                 drop_cls=True):
        super(ReViT, self).__init__()

        with open("/home/workspace/DATA/ReViT/ReViT-b_config.yaml", 'r') as stream:
            cfg_ = yaml.safe_load(stream=stream)
        cfg_ = DefaultMunch.fromDict(cfg_)

        if cfg_.MODEL.name == "ReMViTv2":
            model = build_ReViT(cfg_)
            # load_checkpoint(model, cfg_, "cpu")
        else:
            model = VisionTransformer(
                img_size=224,
                patch_size= 16,
                in_chans = 3,
                num_classes = 0,
                global_pool = 'token',
                embed_dim = 768,
                depth = 12,
                num_heads = 12,
                mlp_ratio = 4.,
                qkv_bias = True,
                qk_norm = False,
                class_token = True,
                pre_norm = False,
                fc_norm= False,
                drop_rate = 0.2,
                patch_drop_rate = 0.,
                proj_drop_rate = 0.2,
                attn_drop_rate = 0.2,
                drop_path_rate = 0.2,
                norm_layer= nn.LayerNorm,
                act_layer = nn.GELU,
            )

        self.model = model

    def forward(self, video, batch_size):
        video_flat = video.reshape(video.shape[0], -1, video.shape[-2], video.shape[-1])
        return self.model(video_flat, batch_size)




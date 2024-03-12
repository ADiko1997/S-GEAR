"""
Model architectures.
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

__all__ = [
    'r2plus1d_34',
    'r2plus1d_152',
    'ir_csn_152',
    'ip_csn_152',
    'ip_csn_50',
    # 'BNInceptionVideo',
]


class BasicStem_Pool(nn.Sequential):
    def __init__(self):
        super(BasicStem_Pool, self).__init__(
            nn.Conv3d(
                3,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3),
                         stride=(1, 2, 2),
                         padding=(0, 1, 1)),
        )


class Conv3DDepthwise(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        assert in_planes == out_planes
        super(Conv3DDepthwise, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            groups=in_planes,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class IPConv3DDepthwise(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        assert in_planes == out_planes
        super(IPConv3DDepthwise, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_planes),
            # nn.ReLU(inplace=True),
            Conv3DDepthwise(out_planes, out_planes, None, stride),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        midplanes = (in_planes * out_planes * 3 * 3 *
                     3) // (in_planes * 3 * 3 + 3 * out_planes)
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


def _set_bn_params(model, bn_eps=1e-3, bn_mom=0.1):
    """
    Set the BN parameters to the defaults: Du's models were trained
        with 1e-3 and 0.9 for eps and momentum resp.
        Ref: https://github.com/facebookresearch/VMZ/blob/f4089e2164f67a98bc5bed4f97dc722bdbcd268e/lib/models/r3d_model.py#L208
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eps = bn_eps
            module.momentum = bn_mom


def r2plus1d_34(pretrained=False,
                progress=False,
                bn_eps=1e-3,
                bn_mom=0.1,
                **kwargs):
    model = _video_resnet("r2plus1d_34",
                          False,
                          False,
                          block=BasicBlock,
                          conv_makers=[Conv2Plus1D] * 4,
                          layers=[3, 4, 6, 3],
                          stem=R2Plus1dStem,
                          **kwargs)
    _set_bn_params(model, bn_eps, bn_mom)
    return model


def r2plus1d_152(pretrained=False,
                 progress=False,
                 bn_eps=1e-3,
                 bn_mom=0.1,
                 **kwargs):
    model = _video_resnet("r2plus1d_152",
                          False,
                          False,
                          block=Bottleneck,
                          conv_makers=[Conv2Plus1D] * 4,
                          layers=[3, 8, 36, 3],
                          stem=R2Plus1dStem,
                          **kwargs)
    _set_bn_params(model, bn_eps, bn_mom)
    return model


def ir_csn_152(pretrained=False,
               progress=False,
               bn_eps=1e-3,
               bn_mom=0.1,
               **kwargs):
    model = _video_resnet("ir_csn_152",
                          False,
                          False,
                          block=Bottleneck,
                          conv_makers=[Conv3DDepthwise] * 4,
                          layers=[3, 8, 36, 3],
                          stem=BasicStem_Pool,
                          **kwargs)
    _set_bn_params(model, bn_eps, bn_mom)
    return model


def ip_csn_152(pretrained=False,
               progress=False,
               bn_eps=1e-3,
               bn_mom=0.1,
               **kwargs):
    model = _video_resnet("ip_csn_152",
                          False,
                          False,
                          block=Bottleneck,
                          conv_makers=[IPConv3DDepthwise] * 4,
                          layers=[3, 8, 36, 3],
                          stem=BasicStem_Pool,
                          **kwargs)
    _set_bn_params(model, bn_eps, bn_mom)
    return model


def ip_csn_50(pretrained=False,
              progress=False,
              bn_eps=0.3,
              bn_mom=0.1,
              **kwargs):
    model = _video_resnet("ip_csn_50",
                          False,
                          False,
                          block=Bottleneck,
                          conv_makers=[IPConv3DDepthwise] * 4,
                          layers=[3, 8, 6, 3],
                          stem=BasicStem_Pool,
                          **kwargs)
    _set_bn_params(model, bn_eps, bn_mom)
    return model


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
                #  model_type='vit_base_patch16_clip_224',
                #  model_type='vit_large_patch16_224',
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
                #  model_type='vit_base_patch16_clip_224',
                #  model_type='vit_large_patch16_224',
                 drop_cls=True):
        super(ResNet101, self).__init__()
        model = resnet50()
        self.model = model
        self.model.fc = nn.Identity() 
        # self.model.forward = MethodType(forward_, self.model)

    def forward(self, video, batch_size):
        # batch_size = video.size(0)
        # time_dim = video.size(2)
        video_flat = video.transpose(1, 2).flatten(0, 1)
        return self.model(video_flat)

# class XCLIP(nn.Module):
#     def __init__(self,
#                  num_classes,
#                  model_type='xclip_frame8',
#                 #  model_type='vit_base_patch16_clip_224',
#                 #  model_type='vit_large_patch16_224',
#                  drop_cls=True):
#         super(XCLIP, self).__init__()

#         __clip_confg = {
#         "_name_or_path": "microsoft/xclip-base-patch16-kinetics-600",
#         "attention_dropout": 0.0,
#         "drop_path_rate": 0.0,
#         "dropout": 0.0,
#         "hidden_act": "quick_gelu",
#         "hidden_size": 768,
#         "image_size": 224,
#         "initializer_factor": 1.0,
#         "initializer_range": 0.02,
#         "intermediate_size": 3072,
#         "layer_norm_eps": 1e-05,
#         "mit_hidden_size": 512,
#         "mit_intermediate_size": 2048,
#         "mit_num_attention_heads": 8,
#         "mit_num_hidden_layers": 1,
#         "model_type": "xclip_vision_model",
#         "num_attention_heads": 12,
#         "num_channels": 3,
#         "num_frames": 8,
#         "num_hidden_layers": 12,
#         "patch_size": 16,
#         "transformers_version": "4.31.0"
#         }
        
#         conf = XCLIPVisionConfig(**__clip_confg)
#         model = XCLIPVisionModel(conf)
#         self.model = model
#         # self.model.forward = MethodType(forward_, self.model)

#     def forward(self, video, batch_size):
#         # batch_size = video.size(0)
#         # time_dim = video.size(2)
#         video_flat = video.transpose(1, 2).flatten(0, 1)
#         return self.model(video_flat).last_hidden_state


# class Vivit(nn.Module):
#     def __init__(self, num_classes):
#         super(Vivit, self).__init__()
#         cfg = VivitConfig(num_frames=32)
#         model = VivitModel(cfg)
#         self.model = model
#         self.tubelet_size = cfg.tubelet_size

#         # self.model.forward = MethodType(forward_, self.model)

#     def forward(self, video, batch_size):
#         video = video.squeeze()
#         N, C, W, H = video.shape
#         batch_size = batch_size
#         time_dim = int(N/batch_size)
#         video = video.view(batch_size, time_dim, C, W, H) #B, T, C, W, H
#         output = self.model(video)
#         logits = output.last_hidden_state[:, 1:, :] #B, tokens, dim
#         tmp_slices = int(time_dim/self.tubelet_size[0])
#         logits = logits.reshape(batch_size * tmp_slices, -1, logits.shape[-1]) #B * Tubelet_tmp, num_tokens, dim
#         return logits
        

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



# ONLY FOR Timeshift


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
        # print(video_flat.shape)
        return self.model(video_flat, batch_size)




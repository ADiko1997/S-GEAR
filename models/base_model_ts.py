# Copyright (c) Facebook, Inc. and its affiliates.

"""
The overall base model.
"""
from typing import Dict, Tuple
import operator
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf
from models.attention import ProtAttn, SelfAttention, TimeShiftBlock
from models.common import TopSimilarTokens

CLS_MAP_PREFIX = 'cls_map_'
PAST_LOGITS_PREFIX = 'past_'


class BaseModel(nn.Module):
    def __init__(self, model_cfg: OmegaConf, num_classes: Dict[str, int],
                 class_mappings: Dict[Tuple[str, str], torch.FloatTensor],
                 mod_embeddings:torch.FloatTensor,
                 text_embeddings:torch.FloatTensor):
        super().__init__()
        
        # Takes as input (B, T, H, W, C) -> (B, T', H', W', C')
        _backbone_full = hydra.utils.instantiate(
            model_cfg.backbone, num_classes=1)  # Add dummy value for num_cls  # will be removed next anyway

        if model_cfg.backbone_last_n_modules_to_drop > 0:
            self.backbone = nn.Sequential()
            for name, child in list(_backbone_full.named_children(
            ))[:-model_cfg.backbone_last_n_modules_to_drop]:
                self.backbone.add_module(name, child)

        else:
            self.backbone = _backbone_full


        # for name, param in self.backbone.named_parameters():
        #     param.requires_grad = False
        # Map the (B, T', H', W', C') -> (B, T', H', W', C*)
        # to the intermediate feature dimensions
        # IMP: this is only used if C' != C*

        if (model_cfg.backbone_last_n_modules_to_drop == 0
                and 'output_dim' in dir(self.backbone)):
            backbone_dim = self.backbone.output_dim
        else:
            backbone_dim = model_cfg.backbone_dim  # TODO: Figure automatically


        if model_cfg.freeze_weights:
            self.backbone.requires_grad_(requires_grad=False)
            
        if model_cfg.model_name.strip() == 'vivit': #spatio-temporal tokens
            self.tubelet_size = model_cfg.tubelet_size

        else:
            self.tubelet_size = 1

        self.cls_token = model_cfg.cls_token
        #Build TimeShift Module
        self.TimeShift = None
        if model_cfg.backbone_add_ts_blocks != 0:
            self.TimeShift = nn.ModuleList()
            for _ in range(model_cfg.backbone_add_ts_blocks):
                self.TimeShift.append(
                    TimeShiftBlock(
                        dim = model_cfg.backbone_dim,
                        num_heads = model_cfg.backbone_dim // 64,
                        num_tmp_tokens = int(model_cfg.tmp_frames/self.tubelet_size)
                    )
                )

        if model_cfg.prot_attn:        
            #Create modules for CrosModalityAttention
            self.ProtAttn = ProtAttn(dim=model_cfg.intermediate_featdim, num_heads=model_cfg.intermediate_featdim // 64, num_tmp_tokens=int(model_cfg.tmp_frames/self.tubelet_size))
        else:
            self.ProtAttn = nn.Identity()

        # if mod_embeddings:
        self.mod_embeddings = mod_embeddings.requires_grad_(requires_grad=False)
        self.text_embeddings = text_embeddings.requires_grad_(requires_grad=False)
        self.future_steps = model_cfg.future_steps
        # self.vis_prototypes = nn.Parameter(data=self.mod_embeddings.data, requires_grad=(not model_cfg.greeze_prototypes))

        #This prevents to downsize CSN features
        if model_cfg.intermediate_featdim != self.mod_embeddings.size(-1):
            # print("embeddings shape, ", self.mod_embeddings.shape)
            N, C = self.mod_embeddings.size()
            target_shape = (N, model_cfg.intermediate_featdim)
            self.mod_embeddings = self.mod_embeddings.view(1, 1, N, C)
            self.mod_embeddings = torch.nn.functional.interpolate(self.mod_embeddings, size=target_shape, mode='bilinear', align_corners=False).squeeze()
        self.vis_prototypes = nn.Parameter(data=self.mod_embeddings.data, requires_grad=(not model_cfg.greeze_prototypes))


        self.sim_func = nn.CosineSimilarity(dim=-1)
        # self.scaler = nn.Sigmoid()
        self.balancer = nn.Parameter(data=torch.tensor([0.5]), requires_grad=True)
        self.top_similar_fn = TopSimilarTokens()
        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.temperature_scaling = nn.Sigmoid()

        if model_cfg.reg:
            self.reg_features = model_cfg.reg
        else:
            self.reg_features = False            

        self.mapper_to_inter = None
        if model_cfg.intermediate_featdim is None:
            model_cfg.intermediate_featdim = backbone_dim
        if backbone_dim != model_cfg.intermediate_featdim:
            self.mapper_to_inter = nn.Linear(backbone_dim,
                                             model_cfg.intermediate_featdim,
                                             bias=False)


        # Takes as input (B, T', H', W', C*) -> (B, C**)
        self.temporal_aggregator = hydra.utils.instantiate(
            model_cfg.temporal_aggregator,
            in_features=model_cfg.intermediate_featdim)


        self.reset_temp_agg_feat_dim = nn.Sequential()
        temp_agg_output_dim = self.temporal_aggregator.output_dim
        if model_cfg.same_temp_agg_dim and (temp_agg_output_dim !=
                                            model_cfg.intermediate_featdim):
            # Ideally want to maintain it so that the same project_mlp
            # can be used for the temporally aggregated features, or the
            # original features.
            self.reset_temp_agg_feat_dim = nn.Linear(
                temp_agg_output_dim, model_cfg.intermediate_featdim)
            temp_agg_output_dim = model_cfg.intermediate_featdim


        # Transforms the current features to future ones
        # (B, C**) -> (B, C**)
        self.future_predictor = hydra.utils.instantiate(
            model_cfg.future_predictor,
            in_features=temp_agg_output_dim,
            _recursive_=False)
        
        if model_cfg.freeze_future_predictor:
            self.future_predictor.requires_grad_(requires_grad=False)
        

        # Projection layer
        self.project_mlp = nn.Sequential()
        if model_cfg.project_dim_for_nce is not None:
            self.project_mlp = nn.Sequential(
                nn.Linear(temp_agg_output_dim, temp_agg_output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(temp_agg_output_dim, model_cfg.project_dim_for_nce))


        # 2nd round of temporal aggregation, if needed
        self.temporal_aggregator_after_future_pred = hydra.utils.instantiate(
            model_cfg.temporal_aggregator_after_future_pred,
            self.future_predictor.output_dim)


        # Dropout
        self.dropout = nn.Dropout(model_cfg.dropout)


        # Takes as input (B, C**) -> (B, num_classes)
        cls_input_dim = self.temporal_aggregator_after_future_pred.output_dim
        # Make a separate classifier for each output
        self.classifiers = nn.ModuleDict()
        self.num_classes = num_classes
        for i, (cls_type, cls_dim) in enumerate(num_classes.items()):
            if model_cfg.use_cls_mappings and i > 0:
                # In this case, rely on the class mappings to generate the
                # other predictions, rather than creating a new linear layer
                break

            self.classifiers.update({
                cls_type:
                hydra.utils.instantiate(model_cfg.classifier,
                                        in_features=cls_input_dim,
                                        out_features=cls_dim)
            })
            # self.classifiers[cls_type].requires_grad_(requires_grad=model_cfg.train_cls)

        # Store the class mappings as buffers
        for (src, dst), mapping in class_mappings.items():
            self.register_buffer(f'{CLS_MAP_PREFIX}{src}_{dst}', mapping)
            # print(f'{CLS_MAP_PREFIX}{src}_{dst}')
        

        self.regression_head = None
        if model_cfg.add_regression_head:
            self.regression_head = nn.Linear(cls_input_dim, 1)

        if model_cfg.multimodal:
            self.multimodal_attn = SelfAttention(
                    dim=model_cfg.intermediate_featdim, 
                    num_heads=8, qkv_bias=True, 
                    attn_drop=0.2, proj_drop=0.2
                    )
            self.modality_balancer = nn.Parameter(data=torch.tensor([0.5]), requires_grad=True)
            


        if model_cfg.extracted_features and model_cfg.use_object:
            self.transform = nn.Linear(352, model_cfg.intermediate_featdim)

        if model_cfg.freeze_future_predictor:
            self.future_predictor.requires_grad_(requires_grad=False)
        
        if model_cfg.transform_features:
            self.transform_features = nn.Sequential(
                nn.Linear(model_cfg.intermediate_featdim, model_cfg.intermediate_featdim),
                nn.LayerNorm(model_cfg.intermediate_featdim),
                nn.ReLU(inplace=True),
                nn.Linear(model_cfg.intermediate_featdim, model_cfg.intermediate_featdim)
            )
            if model_cfg.freeze_transform_features:
                self.transform_features.requires_grad_(requires_grad=False)

        # Init weights, as per the video resnets
        self._initialize_weights()


        # Set he BN momentum and eps here, Du uses a different value and its imp
        self._set_bn_params(model_cfg.bn.eps, model_cfg.bn.mom)
        self.cfg = model_cfg
        self.future_steps = 4

    def _initialize_weights(self):
        # Copied over from
        # https://github.com/pytorch/vision/blob/75f5b57e680549d012b3fc01b356b2fb92658ea7/torchvision/models/video/resnet.py#L261
        # Making sure all layers get init to good video defaults
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _set_bn_params(self, bn_eps=1e-3, bn_mom=0.1):
        """
        Set the BN parameters to the defaults: Du's models were trained
            with 1e-3 and 0.9 for eps and momentum resp.
            Ref: https://github.com/facebookresearch/VMZ/blob/f4089e2164f67a98bc5bed4f97dc722bdbcd268e/lib/models/r3d_model.py#L208
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eps = bn_eps
                module.momentum = bn_mom


    def forward_singlecrop(self, video, target_shape=None):
        """
        Args:
            video (torch.Tensor, Bx#clipsxCxTxHxW)
            target_shape: The shape of the target. Some of these layers might
                be able to use this information.
        """
        outputs = {}
        aux_losses = {}
        batch_size = video.size(0)
        num_clips = video.size(1)
        div = torch.sqrt(torch.tensor(self.cfg.backbone_dim)).to(video.device)
        # Fold the clips dimension into the batch for feature extraction, upto
        # temporal aggregation
        video = video.flatten(0, 1)
        try:
            feats = self.backbone(video)
        except:
            feats = self.backbone(video, batch_size)

        


        outputs['backbone'] = feats.squeeze()    
        # print(f"backbone shape: {outputs['backbone'].shape}")

        # Apply the timeshift module
        if self.TimeShift:
            attn_res = None
            for blk in self.TimeShift:
                feats, attn_res = blk(feats, batch_size, attn_res)

        if self.cls_token:
            feats_agg = feats[:, 0, :]
            vis_feats = outputs['backbone'][:, 0] #visual features used for the prototype attention
        else:
            feats_agg = feats
            vis_feats = feats
        outputs['backbone_mean'] = torch.mean(feats, [-1])

        # If it's not sequential and can be applied here
        if len(self.project_mlp) > 0 and (outputs['backbone_mean'].size(-1) ==
                                          self.project_mlp[0].in_features):
            outputs['backbone_mean_projected'] = self.project_mlp(
                outputs['backbone_mean'])


        # Map the feats to intermediate dimension, that rest of the code
        # will operate on. Only if the original feature is not already
        if feats_agg.shape[-1] != self.cfg.intermediate_featdim:
            assert self.mapper_to_inter is not None, (
                f'The backbone feat does not match intermediate {feats_agg.shape} '
                f'and {self.cfg.intermediate_featdim}. Please set '
                f'model.backbone_dim correctly.')
            feats_agg = self.mapper_to_inter(feats_agg)
            vis_feats = self.mapper_to_inter(vis_feats)


        outputs['temp_agg_projected'] = self.project_mlp(feats_agg)


        # Now before future prediction, need to unfold the clips back out,
        # and concat on the temporal dimension
        if num_clips > 1:
            assert num_clips % self.tubelet_size == 0, f"number of clips {num_clips} shuld be a multiple of {self.tubelet_size}"
            assert (
                (feats_agg.ndim == 2)
                or (feats_agg.ndim == 3 and feats_agg.size(1) == 1)
            ), ('Should be using some temporal aggregation when using clips')
            feats_agg = feats_agg.reshape((batch_size, int(num_clips/self.tubelet_size)) +
                                          feats_agg.shape[1:])
            if feats_agg.ndim == 4:
                feats_agg = torch.flatten(feats_agg, 1, 2)
            # now feats_agg back to 3D (B, T, F)

        feats_past = feats_agg
        if self.cfg.prot_attn:
            vis_feats = vis_feats.reshape(feats_past.shape)
            similarities_past = self.sim_func(vis_feats.unsqueeze(2), self.vis_prototypes)
            sim_tokens = self.top_similar_fn(vis_feats.mean(dim=1, keepdim=False), self.vis_prototypes.detach(), int(num_clips/self.tubelet_size), self.text_embeddings)
            feats_past = self.temperature_scaling(self.alpha)*feats_past + (1-self.temperature_scaling(self.alpha))*self.ProtAttn(sim_tokens, feats_past, batch_size)

        outputs['past_mean_representation'] = feats_past.mean(dim=1, keepdim=False)

        # Now the future prediction, also it might update the past features
        (feats_past, feats_future, future_losses,
         endpoints) = self.future_predictor(feats_past, target_shape)
        aux_losses.update(future_losses)

        #Autoregressive prediction
        if not self.train:
            feats_past_ = feats_past
            for i in range(self.future_steps):
                feats_past_ = torch.stack([feats_past_, feats_future.unsqueeze(dim=1)], dim=1)
                (feats_past_, feats_future_, _,
                _) = self.future_predictor(feats_past_, target_shape)
                feats_future[:, -1] = feats_future_[:, -1]

        if not self.cfg.prot_attn:
            similarities_past = self.sim_func(feats_past.unsqueeze(2), self.vis_prototypes)
            past_similaritie = similarities_past
        else:
            past_similaritie = self.sim_func(feats_past.unsqueeze(2), self.vis_prototypes) #for classification of past instances we want the output of the temporal reasoner

        
        
        #Cosine attention on future features and prototype tokens
        outputs['future_z'] = feats_future #(B, C**) important for regularization loss
        similarity = self.sim_func(feats_future.unsqueeze(1), self.vis_prototypes)
        feats_future_ = (similarity/div).softmax(dim=-1)@self.vis_prototypes.detach()
        if self.cfg.prot_attn:
            feats_future = self.temperature_scaling(self.balancer)*feats_future + (1-self.temperature_scaling(self.balancer))*feats_future_
        # print(f"past features: {feats_past.shape}")
            
        outputs.update(endpoints)
        outputs['similarities'] = similarity
        outputs['similarities_past'] = similarities_past
        outputs['future'] = feats_future
        outputs['past'] = feats_past


        # Apply a classifier on the past features, might be supervising that
        if self.cfg.classifier_on_past:
            feats_past_ = (past_similaritie/div).softmax(dim=-1)@self.vis_prototypes.detach()
            feats_past = self.temperature_scaling(self.balancer)*feats_past + (1-self.temperature_scaling(self.balancer))*feats_past_
            feats_past_drop = self.dropout(feats_past)
            outputs.update(
                self._apply_classifier(feats_past_drop,
                                       outputs_prefix=PAST_LOGITS_PREFIX))


        # For the contrastive loss, I need a projected version of this feature
        outputs['future_projected'] = self.project_mlp(feats_agg)


        # Aggregate again, if asked for
        if isinstance(self.temporal_aggregator_after_future_pred, nn.Identity):
            feats_future_agg, future_agg_losses = (
                self.temporal_aggregator_after_future_pred(feats_future))
        else:
            feats_future_agg, future_agg_losses = (
                self.temporal_aggregator_after_future_pred(feats_future.unsqueeze(1)))

        aux_losses.update(future_agg_losses)
        outputs['future_agg'] = feats_future_agg #(B, C**)


        feats_future_agg_drop = self.dropout(feats_future_agg)
        outputs.update(self._apply_classifier(feats_future_agg_drop))
        if self.regression_head:
            outputs['logits_regression'] = self.regression_head(
                feats_future_agg_drop)
        

        return outputs, aux_losses


    def forward_singlecrop_multimodal(self, video, target_shape=None):
        """
        Args:
            video (torch.Tensor, Bx#clipsxCxTxHxW)
            target_shape: The shape of the target. Some of these layers might
                be able to use this information.
        """
        outputs = {}
        aux_losses = {}

        #separete multimodal features from raw videos
        feats_rulstm = video[1].squeeze()
        video = video[0]

        # print(f"feats_rulstm shape: {feats_rulstm[:, :, 2048:].shape}")
        # print(f"video shape: {video.shape}")
        if self.cfg.use_object:

            if feats_rulstm.ndim == 2:
                feats_rulstm = feats_rulstm.unsqueeze(dim=0)

            if self.cfg.num_modalities == 3: #rgb flow and object
                obj = feats_rulstm[:, :, 2048:] #object features
                rgb_flow = feats_rulstm[:, :, :2048] #separate rgb and flow features
                obj = self.transform(obj) #transform object features to the same dimension as rgb and flow
                feats_rulstm = torch.cat([rgb_flow, obj], dim=-1) #concatenate rgb and object features
            else:
                obj = feats_rulstm[:, :, 1024:] #object features
                rgb_flow = feats_rulstm[:, :, :1024] #separate rgb and flow features
                obj = self.transform(obj) #transform object features to the same dimension as rgb and flow
                feats_rulstm = torch.cat([rgb_flow, obj], dim=-1) #concatenate rgb and object features


        batch_size = video.size(0)
        num_clips = video.size(1)

        # Fold the clips dimension into the batch for feature extraction, upto
        # temporal aggregation
        video = video.flatten(0, 1)
        try:
            feats = self.backbone(video)
        except:
            feats = self.backbone(video, batch_size)

        


        outputs['backbone'] = feats.squeeze()    

        # Apply the timeshift module
        if self.TimeShift:
            attn_res = None
            for blk in self.TimeShift:
                feats, attn_res = blk(feats, batch_size, attn_res)

        if self.cls_token:
            feats_agg = feats[:, 0, :]
            vis_feats = outputs['backbone'][:, 0]
        else:
            feats_agg = feats
            vis_feats = feats
        outputs['backbone_mean'] = torch.mean(feats, [-1])

        # If it's not sequential and can be applied here
        if len(self.project_mlp) > 0 and (outputs['backbone_mean'].size(-1) ==
                                          self.project_mlp[0].in_features):
            outputs['backbone_mean_projected'] = self.project_mlp(
                outputs['backbone_mean'])


        # Map the feats to intermediate dimension, that rest of the code
        # will operate on. Only if the original feature is not already
        if feats_agg.shape[-1] != self.cfg.intermediate_featdim:
            assert self.mapper_to_inter is not None, (
                f'The backbone feat does not match intermediate {feats_agg.shape} '
                f'and {self.cfg.intermediate_featdim}. Please set '
                f'model.backbone_dim correctly.')
            feats_agg = self.mapper_to_inter(feats_agg)
            vis_feats = self.mapper_to_inter(vis_feats)


        outputs['temp_agg_projected'] = self.project_mlp(feats_agg)


        # Now before future prediction, need to unfold the clips back out,
        # and concat on the temporal dimension
        if num_clips > 1:
            assert num_clips % self.tubelet_size == 0, f"number of clips {num_clips} shuld be a multiple of {self.tubelet_size}"
            assert (
                (feats_agg.ndim == 2)
                or (feats_agg.ndim == 3 and feats_agg.size(1) == 1)
            ), ('Should be using some temporal aggregation when using clips')
            feats_agg = feats_agg.reshape((batch_size, int(num_clips/self.tubelet_size)) +
                                          feats_agg.shape[1:])
            if feats_agg.ndim == 4:
                feats_agg = torch.flatten(feats_agg, 1, 2)
            # now feats_agg back to 3D (B, T, F)

        feats_past = feats_agg
        if self.cfg.multimodal:
            rulstm_feats = feats_rulstm.reshape(feats_rulstm.shape[0], feats_rulstm.shape[1], -1, feats_past.shape[-1])
            multimodal_feats = torch.cat([feats_past.unsqueeze(2), rulstm_feats], dim=2)
            multimodal_feats = self.multimodal_attn(multimodal_feats)
            feats_past = self.temperature_scaling(self.modality_balancer)*multimodal_feats[:, :, 0, :].squeeze(2) + feats_past


        # if batch_size != target_shape[0]: #usefull when having multiple videos of the same label but with different time frames
        outputs['past_mean_representation'] = feats_past.mean(dim=1, keepdim=False) #use this for the trained architectures
        # vis_feats = vis_feats.reshape(feats_past.shape)
        # outputs['similarities_past'] = self.sim_func(vis_feats.mean(dim=1, keepdim=False).unsqueeze(1).detach(), self.vis_prototypes)
        # sim_tokens = self.top_similar_fn(vis_feats.mean(dim=1, keepdim=False), self.vis_prototypes, int(num_clips/self.tubelet_size), self.text_embeddings)
        # feats_past = self.temperature_scaling(self.alpha)*feats_past + (1-self.temperature_scaling(self.alpha))*self.ProtAttn(sim_tokens, feats_past, batch_size)



        # Now the future prediction, also it might update the past features
        (feats_past, feats_future, future_losses,
         endpoints) = self.future_predictor(feats_past, target_shape)
        aux_losses.update(future_losses)


        #Autoregressive prediction
        if not self.train:
            feats_past_ = feats_past
            for i in range(self.future_steps):
                feats_past_ = torch.stack([feats_past_, feats_future.unsqueeze(dim=1)], dim=1)
                (feats_past_, feats_future_, _,
                _) = self.future_predictor(feats_past_, target_shape)
                feats_future[:, -1] = feats_future_[:, -1]

        if not self.cfg.prot_attn:
            similarities_past = self.sim_func(feats_past.unsqueeze(2).detach(), self.vis_prototypes)
            past_similaritie = similarities_past
        else:
            past_similaritie = self.sim_func(feats_past.unsqueeze(2), self.vis_prototypes) #for classification of past instances we want the output of the temporal reasoner

        
        
        #Cosine attention on future features and prototype tokens
        similarity = self.sim_func(feats_future.unsqueeze(1).detach(), self.vis_prototypes)
        feats_future_ = (similarity/div).softmax(dim=-1)@self.vis_prototypes.detach()
        feats_future = self.temperature_scaling(self.balancer)*feats_future + (1-self.temperature_scaling(self.balancer))*feats_future_
        # print(f"past features: {feats_past.shape}")
            
        outputs.update(endpoints)
        outputs['similarities'] = similarity
        outputs['similarities_past'] = similarities_past
        outputs['future'] = feats_future
        outputs['past'] = feats_past


        # Apply a classifier on the past features, might be supervising that
        if self.cfg.classifier_on_past:
            feats_past_ = (past_similaritie/div).softmax(dim=-1)@self.vis_prototypes.detach()
            feats_past = self.temperature_scaling(self.balancer)*feats_past + (1-self.temperature_scaling(self.balancer))*feats_past_
            feats_past_drop = self.dropout(feats_past)
            outputs.update(
                self._apply_classifier(feats_past_drop,
                                       outputs_prefix=PAST_LOGITS_PREFIX))


        # For the contrastive loss, I need a projected version of this feature
        outputs['future_projected'] = self.project_mlp(feats_agg)


        # Aggregate again, if asked for
        if isinstance(self.temporal_aggregator_after_future_pred, nn.Identity):
            feats_future_agg, future_agg_losses = (
                self.temporal_aggregator_after_future_pred(feats_future))
        else:
            feats_future_agg, future_agg_losses = (
                self.temporal_aggregator_after_future_pred(feats_future.unsqueeze(1)))

        aux_losses.update(future_agg_losses)
        outputs['future_agg'] = feats_future_agg #(B, C**)


        feats_future_agg_drop = self.dropout(feats_future_agg)
        outputs.update(self._apply_classifier(feats_future_agg_drop))
        if self.regression_head:
            outputs['logits_regression'] = self.regression_head(
                feats_future_agg_drop)
        

        return outputs, aux_losses


    def forward_singlecrop_features(self, video, target_shape=None):
        """
        Args:
            video (torch.Tensor, Bx#clipsxCxTxHxW)
            target_shape: The shape of the target. Some of these layers might
                be able to use this information.
        """
        outputs = {}
        aux_losses = {}
        
        video = video.squeeze()
        batch_size = video.size(0)
        num_clips = video.size(1)
        feats_past = video
        div = torch.sqrt(torch.tensor(self.cfg.backbone_dim)).to(video.device)

        if feats_past.shape[-1] != self.cfg.intermediate_featdim:
            assert self.mapper_to_inter is not None, (
                f'The backbone feat does not match intermediate {feats_past.shape} '
                f'and {self.cfg.intermediate_featdim}. Please set '
                f'model.backbone_dim correctly.')
            feats_past = self.mapper_to_inter(feats_past)

        #Apply standard scaler to match the prototype distribution
        if self.cfg.prot_attn: #standardize features to match the prototype distribution when we use prototype attention
            feats_past = self.standard_scaler(feats_past, self.vis_prototypes.detach())

        
        if self.cfg.transform_features:
            feats_past = feats_past + self.transform_features(feats_past)
            # feats_past = self.transform_features(feats_past)

        #Apply multimodal attention in case of multimodal extracted features
        if self.cfg.multimodal:
            feats_past = self.multimodal_attention(None, feats_past)


        if self.train and self.cfg.feature_mixup:
            #mixup features
            feats_past_ = torch.flip(feats_past, dims=[0]).detach()
            random_number = (torch.rand(1) * 0.3).to(feats_past.device)
            feats_past = feats_past + random_number*feats_past_ #mixup noise

        outputs['past_mean_representation'] = feats_past.mean(dim=1, keepdim=False) #use this for the trained architectures
        if self.cfg.prot_attn:
            vis_feats = feats_past
            similarities_past = self.sim_func(vis_feats.unsqueeze(2), self.vis_prototypes.detach())
            sim_tokens = self.top_similar_fn(vis_feats.mean(dim=1, keepdim=False), self.vis_prototypes.detach(), int(num_clips/self.tubelet_size))
            feats_past = self.temperature_scaling(self.alpha)*feats_past + (1-self.temperature_scaling(self.alpha))*self.ProtAttn(sim_tokens, feats_past, batch_size)


        (feats_past, feats_future, future_losses,
         endpoints) = self.future_predictor(feats_past, target_shape)
        aux_losses.update(future_losses)

        if not self.cfg.prot_attn:
            similarities_past = self.sim_func(feats_past.unsqueeze(2).detach(), self.vis_prototypes)
            past_similaritie = similarities_past
        else:
            past_similaritie = self.sim_func(feats_past.unsqueeze(2), self.vis_prototypes) #for classification of past instances we want the output of the temporal reasoner

        
        
        #Cosine attention on future features and prototype tokens
        outputs['future_z'] = feats_future #(B, C**) important for regularization loss
        similarity = self.sim_func(feats_future.unsqueeze(1).detach(), self.vis_prototypes)
        feats_future_ = (similarity/div).softmax(dim=-1)@self.vis_prototypes
        feats_future = self.temperature_scaling(self.balancer)*feats_future + (1-self.temperature_scaling(self.balancer))*feats_future_
        # print(f"past features: {feats_past.shape}")
            
        outputs.update(endpoints)
        outputs['similarities'] = similarity
        outputs['similarities_past'] = similarities_past
        outputs['future'] = feats_future
        outputs['past'] = feats_past


        # Apply a classifier on the past features, might be supervising that
        if self.cfg.classifier_on_past:
            feats_past_ = (past_similaritie/div).softmax(dim=-1)@self.vis_prototypes
            feats_past = self.temperature_scaling(self.balancer)*feats_past + (1-self.temperature_scaling(self.balancer))*feats_past_
            feats_past_drop = self.dropout(feats_past)
            outputs.update(
                self._apply_classifier(feats_past_drop,
                                       outputs_prefix=PAST_LOGITS_PREFIX))


        # For the contrastive loss, I need a projected version of this feature
        outputs['future_projected'] = self.project_mlp(video.squeeze())


        # Aggregate again, if asked for
        if isinstance(self.temporal_aggregator_after_future_pred, nn.Identity):
            feats_future_agg, future_agg_losses = (
                self.temporal_aggregator_after_future_pred(feats_future))
        else:
            feats_future_agg, future_agg_losses = (
                self.temporal_aggregator_after_future_pred(feats_future.unsqueeze(1)))

        aux_losses.update(future_agg_losses)
        # outputs['future_agg'] = feats_future_agg #(B, C**)


        feats_future_agg_drop = self.dropout(feats_future_agg)
        outputs.update(self._apply_classifier(feats_future_agg_drop))
        if self.regression_head:
            outputs['logits_regression'] = self.regression_head(
                feats_future_agg_drop)
        

        return outputs, aux_losses

    def translation(self, feats):
        """
        Translate one dimensional feature vectors to the same dimension as the multimodal features
        Args:
            feats (torch.Tensor, Bx#clipsxD)
        Output:
            feats (torch.Tensor, Bx#clipsxD)
        """
        random_seed = torch.randn(1).to(feats.device)
        feats = feats + random_seed
        return feats


    def gaussian_noise(self, feats):
        """
        Translate one dimensional feature vectors to the same dimension as the multimodal features
        Args:
            feats (torch.Tensor, Bx#clipsxD)
        Output:
            feats (torch.Tensor, Bx#clipsxD)
        """
        random_seed = torch.randn((feats.shape)).to(feats.device)
        feats = feats + random_seed
        return feats

    @staticmethod
    def shift_tensor(tensor, fill_value=0):
        """
        Shifts a 1D tensor by the specified number of positions.
        New entries are filled with `fill_value`.
        Elements that are shifted out of the tensor's bounds are discarded.

        :param tensor: The input tensor.
        :param shift: The number of positions to shift the tensor. A positive value shifts right, a negative value shifts left.
        :param fill_value: The value to fill the new entries with. Default is 0.
        :return: The shifted tensor.
        """
        shift = torch.randint(low=-int(tensor.size(-1)/32), high=int(tensor.size(-1)/32), size=(1,)).item()
        result = torch.full_like(tensor, fill_value)
        if shift > 0:
            result[:,:, shift:] = tensor[:,:,:-shift]
        elif shift < 0:
            result[:,:,:shift] = tensor[:,:,-shift:]
        else:
            result = tensor.clone()
        return result

    @staticmethod
    def shift_time(tensor, fill_value=0):
        """
        Shifts a 1D tensor by the specified number of positions.
        New entries are filled with `fill_value`.
        Elements that are shifted out of the tensor's bounds are discarded.

        :param tensor: The input tensor.
        :param shift: The number of positions to shift the tensor. A positive value shifts right, a negative value shifts left.
        :param fill_value: The value to fill the new entries with. Default is 0.
        :return: The shifted tensor.
        """
        shift = torch.randint(low=-int(tensor.size(1)/5), high=int(tensor.size(1)/5), size=(1,)).item()
        result = torch.full_like(tensor, fill_value)
        if shift > 0:
            result[:, shift:] = tensor[:,:-shift]
        elif shift < 0:
            result[:,:shift] = tensor[:,-shift:]
        else:
            result = tensor.clone()
        return result


    @staticmethod
    def drop_samples(tensor):
        """
        Drops a random number of samples from the tensor
        """
        result = torch.full_like(tensor, 0).to(tensor.device)
        drop = torch.randint(low=0, high=int(tensor.size(0)/4), size=(1,)).item()
        result[drop:] = tensor[drop:]
        return result

    @staticmethod
    def standard_scaler(tensor, dist):
        """
        Standardize features by removing the mean and scaling to unit variance
        """
        result = tensor - dist.mean()
        result = result / dist.std()
        return result


    def _apply_classifier(self, input_feat, outputs_prefix=''):
        outputs = {}
        for key in self.num_classes.keys():
            if key in self.classifiers:
                outputs[f'{outputs_prefix}logits/{key}'] = self.classifiers[
                    key](input_feat)
            else:
                # A mapping must exist, in order to compute this, and must
                # have been computed already (so ordering in the config
                # matters)
                src_key = next(iter(self.classifiers.keys()))
                src_tensor = outputs[f'{outputs_prefix}logits/{src_key}']
                mapper = operator.attrgetter(
                    f'{CLS_MAP_PREFIX}{key}_{src_key}')(self)
                outputs[f'{outputs_prefix}logits/{key}'] = torch.mm(
                    src_tensor, mapper)
        return outputs

    
    def multimodal_attention(self, feats, rulstm_feats):
        #compute multimodal self-attention given the modalities
        if feats is not None:
            if feats.ndim == 2:
                rulstm_feats = rulstm_feats.reshape(feats.shape[0], -1)
            else:
                rulstm_feats = rulstm_feats.reshape(feats.shape[0], feats.shape[1], -1)

            # print(f"feats shape: {feats.shape}")
            # print(f"rulstm_feats shape: {rulstm_feats.shape}")
            
            feats_cat = torch.cat([feats, rulstm_feats], dim=-1)
            # print(f"feats_cat shape: {feats_cat.shape}")
            attn_score = self.multimodal_attn(feats_cat)
            mod_1 = rulstm_feats[:, :, :1024] #mod are concatenated in the last dimension (rgn + flow)
            if self.cfg.num_modalities == 2:
                mod_2 = rulstm_feats[:, :, 1024:]
                attn_score = attn_score.reshape(feats.size(0), feats.size(1), int(feats_cat.size(-1)/feats.size(-1)))
                attn_score = attn_score.softmax(dim=-1)
                feats = attn_score[:, :, 0].unsqueeze(2)*feats + attn_score[:, :, 1].unsqueeze(2)*mod_1 + attn_score[:, :, 2].unsqueeze(2)*mod_2

            elif self.cfg.num_modalities == 3:
                mod_2 = rulstm_feats[:, :, 1024:2048] # flow
                mod_3 = rulstm_feats[:, :, 2048:] # object

                attn_score = attn_score.reshape(feats.size(0), feats.size(1), int(feats_cat.size(-1)/feats.size(-1)))
                attn_score = attn_score.softmax(dim=-1)
                feats = attn_score[:, :, 0].unsqueeze(2)*feats + attn_score[:, :, 1].unsqueeze(2)*mod_1 + attn_score[:, :, 2].unsqueeze(2)*mod_2 + attn_score[:, :, 3].unsqueeze(2)*mod_3

        else:
            feats = rulstm_feats
            attn_score = self.multimodal_attn(feats)
            mod_0 = rulstm_feats[:, :, :1024] #mod are concatenated in the last dimension (rgn + flow)
            if self.cfg.num_modalities == 2:
                mod_1 = rulstm_feats[:, :, 1024:]
                attn_score = attn_score.reshape(feats.size(0), feats.size(1), int(feats_cat.size(-1)/feats.size(-1)))
                attn_score = attn_score.softmax(dim=-1)
                feats = attn_score[:, :, 0].unsqueeze(2)*mod_0 + attn_score[:, :, 1].unsqueeze(2)*mod_1

            elif self.cfg.num_modalities == 3:
                mod_2 = rulstm_feats[:, :, 1024:2048] # flow
                mod_3 = rulstm_feats[:, :, 2048:] # object

                attn_score = attn_score.reshape(feats.size(0), feats.size(1), int(feats_cat.size(-1)/feats.size(-1)))
                attn_score = attn_score.softmax(dim=-1)
                feats = attn_score[:, :, 0].unsqueeze(2)*feats + attn_score[:, :, 1].unsqueeze(2)*mod_1 + attn_score[:, :, 2].unsqueeze(2)*mod_2 + attn_score[:, :, 3].unsqueeze(2)*mod_3


        return feats


    def forward(self, video, epoch, *args, **kwargs):
        """
            Args: video (torch.Tensor)
                Could be (B, #clips, C, T, H, W) or
                    (B, #clips, #crops, C, T, H, W)
            Returns:
                Final features
                And any auxiliarly losses produced by the model
        """
        
        if video['mod_0'].ndim == 6:
            video_crops = [video[mod] for mod in video.keys()]
        elif video['mod_0'].ndim == 7 and video['mod_0'].size(2) == 1:
            video['mod_0'] = video['mod_0'].squeeze(2)
            video_crops = [video[mod] for mod in video.keys()]
        elif video['mod_0'].ndim == 7:
            assert not self.cfg.multimodal, "Multicrop testing is only used for raw rgb videos"
            video_crops = torch.unbind(video['mod_0'], dim=2) #not considerng crops at the moment for multimodal
        else:
            raise NotImplementedError('Unsupported size %s' % video['mod_0'].shape)


        if self.cfg.multimodal:
            feats_losses = [
                self.forward_singlecrop_multimodal(video_crops, *args, **kwargs)
            ]

        elif self.cfg.extracted_features:
            if self.training:
                # video_crops[0] = self.shift_time(video_crops[0], fill_value=0)
                # video_crops[0] = self.drop_samples(video_crops[0])

                feats_losses = [
                    self.forward_singlecrop_features(video_crops[0], *args, **kwargs)
                ]
            else:
                feats_losses = [
                    self.forward_singlecrop_features(video_crops[0], *args, **kwargs)
                ]

        else:
            feats_losses = [
                self.forward_singlecrop(el, *args, **kwargs) for el in video_crops
            ]

        feats, losses = zip(*feats_losses)
        feats = {k: [dic[k] for dic in feats] for k in feats[0]}
        losses = {k: [dic[k] for dic in losses] for k in losses[0]}

        # Average over the crops
        feats = {
            k: torch.mean(torch.stack(el, dim=0), dim=0)
            for k, el in feats.items()
        }

        losses = {
            k: torch.mean(torch.stack(el, dim=0), dim=0)
            for k, el in losses.items()
        }
        return feats, losses

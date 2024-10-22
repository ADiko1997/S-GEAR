# Modified from AVT
"""
Modular implementation of the basic train ops
"""
from typing import Dict, Union, Tuple
import torch
import torch.nn as nn
import hydra
from hydra.types import TargetConf

from common import utils

from datasets.base_video_dataset import FUTURE_PREFIX
from models.base_model import PAST_LOGITS_PREFIX
from loss_fn.multidim_xentropy import MultiDimCrossEntropy
from loss_fn.l1loss import NormalizedL1
from loss_fn.mse import NormedMSE
from loss_fn.info_nce import InfoNCE
from models.common import TopSimilarTokens


class NoLossAccuracy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return {}, {}


class BasicLossAccuracy(nn.Module):
    def __init__(self, dataset, device, balance_classes=False):
        super().__init__()
        kwargs = {'ignore_index': -1}
        if balance_classes:
            assert dataset.class_balanced_sampling is False, (
                'Do not re-weight the losses, and do balanced sampling')
            weight = torch.zeros((len(dataset.classes, )),
                                 device=device,
                                 dtype=torch.float)
            for cls_id, count in dataset.classes_counts.items():
                weight[cls_id] = count
            weight = weight / torch.sum(weight)  # To get ratios for non -1 cls
            weight = 1 / (weight + 0.00001)
            kwargs['weight'] = weight
        kwargs['reduction'] = 'none'
        cont_kwargs = {"reduction": 'none'}  # to get batch level output
        
        self.cls_criterion = MultiDimCrossEntropy(**kwargs)
        self.global_SimContrast = NormalizedL1(**cont_kwargs)
        self.similarity = torch.nn.CosineSimilarity(dim=-1) 
        self.distance = NormedMSE(**cont_kwargs)

    def forward(self, outputs, target, target_subclips, mixup_enabled=False, one_hot=False, prototypes=None, 
                target_similarities=None, past_target_similarities=None,
                past_wts=None):
        """
        Args:
            outputs['logits'] torch.Tensor (B, num_classes) or
                (B, T, num_classes)
                Latter in case of dense prediction
            target: {type: (B) or (B, T')}; latter in case of dense prediction
            target_subclips: {type: (B, #clips, T)}: The target for each input
            mixup_enabled: bool: If mixup is enabled
            one_hot: bool: If the target is one hot encoded
            prototypes: torch.Tensor: (C, D) where C is the number of classes
                and D is the dimensionality of the embedding
            target_similarities: torch.Tensor: (B, C) where C is the number of
                classes
            past_target_similarities: torch.Tensor: (B, C) where C is the number
                of classes
            past_wts: torch.Tensor: (B, 1) weights for the past target
        """
        losses = {}
        accuracies = {}
        weights = None
        for tgt_type, tgt_val in target.items():
            logits = outputs[f'logits/{tgt_type}']

            #Sanity check
            if mixup_enabled:
                assert logits.ndim == tgt_val.ndim 
            else:
                assert logits.ndim == tgt_val.ndim + 1

            loss = self.cls_criterion(logits, tgt_val, one_hot)
            dataset_max_classes = logits.size(-1)
            acc1, acc5 = utils.accuracy(logits,
                                        tgt_val,
                                        topk=(1, min(5, dataset_max_classes)),
                                        mixup_enabled=mixup_enabled)
            # Don't use / in loss since I use the config to set weights, and
            # can't use / there.
            losses[f'cls_{tgt_type}'] = loss
            accuracies[f'acc1/{tgt_type}'] = acc1
            accuracies[f'acc5/{tgt_type}'] = acc5
            # Incur past losses
            past_logits_key = f'{PAST_LOGITS_PREFIX}logits/{tgt_type}'
            # If this key exists, means we asked for classifier on the last
            # layer, so the loss should be incurred.
            if past_logits_key in outputs and target_subclips is not None:
                past_logits = outputs[past_logits_key]
                # Take mode over the frames to get the subclip level loss
                if mixup_enabled:
                    past_target = target_subclips[tgt_type]
                else:
                    past_target = torch.mode(target_subclips[tgt_type], -1)[0]
                    mean_mode = torch.mode(past_target, dim=-1)[0]
                    weights = torch.where(mean_mode > 0, 1, 0)
                    # print(f"mean mode: {mean_mode} weights: {weights}")
                
                    assert past_logits.shape[:-1] == past_target.shape, (
                        f'The subclips should be set such that the past logits '
                        f'and past targets match in shape. Currently they are '
                        f'{past_logits.shape} and {past_target.shape}')

                losses[f'past_cls_{tgt_type}'] = self.cls_criterion(
                    past_logits, past_target, one_hot)
            # Else likely not using subclips, so no way to do this loss

        #Calculate global semantic loss
        if len(target_similarities.shape) == 1: #Single instance batch
            target_similarities = target_similarities.unsqueeze(0)

        if len(outputs['similarities'].shape) == 1: #Single instance batch
            target_similarities = target_similarities.unsqueeze(0)


        indices = tgt_val.view(-1).long()
        target_embeddings = prototypes[indices]
         # distance from prototypes
        losses['global_simContrast'] = self.global_SimContrast(outputs['similarities'], target_similarities.squeeze()).sum(dim=-1) #Semantic Loss
        losses['target_distance'] = self.distance(outputs['future_z'], target_embeddings.detach()).sum(dim=-1) #Regularization Loss
        

        if past_target_similarities is not None and target_subclips is not None:
            pst_lss = self.global_SimContrast(outputs['similarities_past'], past_target_similarities.squeeze()).sum(dim=-1) 
            losses['past_target_global_simContrast'] = pst_lss * past_wts.squeeze()
        return losses, accuracies


class Basic:
    def __init__(self,
                 model,
                 device,
                 dataset,
                 cls_loss_acc_fn: TargetConf,
                 reg_criterion: TargetConf = None,
                 mixup_fn=None,
                 store_embeddings:bool=False):
        super().__init__()

        self.model = model
        self.device = device
        self.cls_loss_acc_fn = hydra.utils.instantiate(cls_loss_acc_fn, dataset, device)
        self.mixup_fn = mixup_fn
        if dataset.visual_embeddings_dict:
            self.class_embeddings = torch.stack(list(dataset.visual_embeddings_dict.values())).to(self.device)
        else:
            self.class_embeddings = torch.stack(list(dataset.embeddings_dict.values())).to(self.device)

        self.classes = torch.Tensor(list(dataset.embeddings_dict.keys())).to(self.device)

        if store_embeddings:
            self.store_embeddings = {}
            for i in range(self.class_embeddings.shape[0]):
                self.store_embeddings[i] = []
        else:
            self.store_embeddings = store_embeddings


        del reg_criterion  # not used here

    def _basic_preproc(self, data, train_mode):
        if not isinstance(data, dict):
            video, target = data
            # Make a dict so that later code can use it
            data = {}
            data['video'] = video
            data['target'] = target
            data['idx'] = -torch.ones_like(target)

        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        return data

    def __call__(
            self,
            data: Union[Dict[str, torch.Tensor],  # If dict
                        Tuple[torch.Tensor, torch.Tensor]],  # vid, target
            train_mode: bool = True,
            epoch: int = None):
        """
        Args:
            data (dict): Dictionary of all the data from the data loader
        """
        data = self._basic_preproc(data, train_mode)
        mixup_enabled = False
        one_hot = False

        if train_mode:
            if self.mixup_fn:
                if 'target_subclips' in data: 
                    data['target_subclips']['action'] = torch.where(data['target_subclips']['action'] == -1, 0, data['target_subclips']['action'])
                    data['video'], data['target_one_hot'], data['target_subclips_one_hot'] = self.mixup_fn(data['video'], data['target']['action'], data['target_subclips']['action'])
                else:
                    data['video'], data['target_one_hot'] = self.mixup_fn(data['video'], data['target']['action'])
                mixup_enabled = True

        video = {mod:data[mod]['video'].to(self.device, non_blocking=True) for mod in data.keys() if 'video' in data[mod]}
        target = {}
        target_subclips = {}

        for key in data['mod_0']['target'].keys():
            if not train_mode or not self.mixup_fn:
                target[key] = data['mod_0']['target'][key].to(self.device, non_blocking=True)
                future_wts = data['mod_0']['target_weights'].to(self.device, non_blocking=True)
            else:
                target[key] = data['mod_0']['target_one_hot'].to(self.device, non_blocking=True)
                future_wts = data['mod_0']['target_weights'].to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            outputs, aux_losses = self.model(video, target_shape=next(iter(target.values())).shape, epoch=epoch)

        if self.store_embeddings:
            for idx, val in enumerate(target['action']):
                if len(self.store_embeddings[val.item()]) < 20:
                    self.store_embeddings[val.item()].append(outputs['future_agg'][idx].clone().detach().tolist())


        if 'target_subclips' in data['mod_0']:
            for key in data['mod_0']['target_subclips'].keys():
                if train_mode and self.mixup_fn:
                    target_subclips[key] = data['mod_0']['target_subclips_one_hot'].to(
                        self.device, non_blocking=True)
                else:
                    target_subclips[key] = data['mod_0']['target_subclips'][key].to(
                    self.device, non_blocking=True)
        else:
            target_subclips = None
            data['target_subclips_ignore_index'] = None

        target_embeddings = data['mod_0']['target_embeddings'].to(self.device) #(B, C**)
        target_similarities = data['mod_0']['target_similarities'].to(self.device)
        if 'target_subclips' in data['mod_0']:
            past_target_similarities = data['mod_0']['past_reference_similarities'].to(self.device)
            past_target_wts = data['mod_0']['past_wts'].to(self.device)
        else:
            past_target_similarities = None
            past_target_wts = None


        with torch.cuda.amp.autocast(enabled=True):
            losses, accuracies = self.cls_loss_acc_fn(outputs, target,
                                                    target_subclips, mixup_enabled, 
                                                    mixup_enabled, self.model.module.org_gamma.data,
                                                    target_similarities, past_target_similarities,
                                                     self.classes, future_wts, past_target_wts) #self.model.org_gamma is the class embeddings
        losses.update(aux_losses)

        return data, outputs, losses, accuracies



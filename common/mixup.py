# """
# Implementation of mixup with ignore class, since some sequences donot have gt labels
# Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
# """

# from typing import Dict, Sequence, Union
# import torch


# def batch_wo_ignore_cls(target_subclips: torch.Tensor, ignore_cls=-1):
#     target_subclips = target_subclips.squeeze(-1)  # avoid dim like (B, 1)
#     assert target_subclips.ndim == 2, "Target subclips should have dimension of 2."
#     batch_index = (target_subclips != ignore_cls).all(-1)
#     return batch_index


# def convert_to_one_hot(
#     targets: torch.Tensor,
#     num_class: int,
#     label_smooth: float = 0.0,
# ) -> torch.Tensor:
#     """
#     This function converts target class indices to one-hot vectors,
#     given the number of classes.
#     Args:
#         targets (torch.Tensor): Index labels to be converted.
#         num_class (int): Total number of classes.
#         label_smooth (float): Label smooth value for non-target classes. Label smooth
#             is disabled by default (0).
#     """
#     assert (
#         torch.max(targets).item() < num_class
#     ), "Class Index must be less than number of classes"
#     assert 0 <= label_smooth < 1.0, "Label smooth value needs to be between 0 and 1."

#     targets = targets.squeeze(-1)  # avoids dim like (B, 1)

#     non_target_value = label_smooth / num_class
#     target_value = 1.0 - label_smooth + non_target_value
#     one_hot_targets = torch.full(
#         (*targets.shape, num_class),
#         non_target_value,
#         dtype=None,
#         device=targets.device,
#     )
#     one_hot_targets.scatter_(-1, targets.unsqueeze(-1), target_value)
#     return one_hot_targets


# def _mix_labels(
#     labels: torch.Tensor,
#     num_classes: int,
#     lam: float = 1.0,
#     label_smoothing: float = 0.0,
#     one_hot: bool = False,
# ):
#     """
#     This function converts class indices to one-hot vectors and mix labels, given the
#     number of classes.
#     Args:
#         labels (torch.Tensor): Class labels.
#         num_classes (int): Total number of classes.
#         lam (float): lamba value for mixing labels.
#         label_smoothing (float): Label smoothing value.
#     """
#     if one_hot:
#         labels1 = labels
#         labels2 = labels.flip(0)
#     else:
#         labels1 = convert_to_one_hot(labels, num_classes, label_smoothing)
#         labels2 = convert_to_one_hot(labels.flip(0), num_classes, label_smoothing)
#     return labels1 * lam + labels2 * (1.0 - lam)


# def _mix(inputs: torch.Tensor, batch_wo_ignore_index: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
#     """
#     mix inputs of specific indexes
#     :param inputs: input tensor
#     :param batch_wo_ignore_index: index of batches where ignore class does occur
#     :param lam: mixup lambda
#     :return: mixed inputs
#     """
#     inputs_selected = inputs[batch_wo_ignore_index]
#     inputs_flipped = inputs_selected.flip(0).mul_(1.0 - lam)
#     inputs_selected.mul_(lam).add_(inputs_flipped)
#     inputs[batch_wo_ignore_index] = inputs_selected
#     return inputs


# class MixUp(torch.nn.Module):
#     """
#     Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
#     """

#     def __init__(
#         self,
#         alpha: float = 1.0,
#         label_smoothing: Dict = 0.0,
#         num_classes: Dict = None,
#         ignore_cls=-1,
#     ) -> None:
#         """
#         This implements MixUp for videos.
#         Args:
#             alpha (float): Mixup alpha value.
#             label_smoothing (dict): Label smoothing value.
#             num_classes (dict, int): Number of total classes.
#             one_hot (bool): whether labels are already in one-hot form
#             ignore_cls (int): class that will not contribute for backpropagation
#         """
#         super().__init__()
#         self.mixup_beta_sampler = torch.distributions.beta.Beta(alpha, alpha)
#         self.label_smoothing = label_smoothing
#         self.num_classes = num_classes
#         self.ignore_cls = ignore_cls

#     def forward(
#         self,
#         x_video: Dict,
#         labels: Dict,
#         labels_subclips: Union[Dict, None],
#     ) -> Sequence[Union[Dict, None]]:
#         """
#         :param x_video: Dict of inputs from different modalities
#         :param labels: Dict of action / (verb, noun) labels
#         :param labels_subclips: Dict of action / (verb, noun) labels for past frames
#         :return: mixed inputs and labels

#         NOTE: Exepts are used to avoid errors with non-dict data that are universal for every type
#         """
#         assert x_video.size(0) > 1, "MixUp cannot be applied to a single instance."
#         batch_wo_ignore_index = [...]

#         # convert labels to one-hot format
#         try:
#             labels_out = {key: convert_to_one_hot(val, self.num_classes[key], self.label_smoothing[key])
#                       for key, val in labels.items()}
#         except:
#             labels_out = {key:  convert_to_one_hot(val, self.num_classes[key], self.label_smoothing) for key, val in labels.items()}

#         if labels_subclips is not None:
#             try:
#                 labels_subclips_out = {key: convert_to_one_hot(val, self.num_classes[key], self.label_smoothing[key])
#                       for key, val in labels_subclips.items()}
#             except:
#                 labels_subclips_out = {key:  convert_to_one_hot(val, self.num_classes[key], self.label_smoothing) for key, val in labels_subclips.items()}

#             # labels_subclips_curr = next(iter(labels_subclips.values()))
            
#             # batch_wo_ignore_index = batch_wo_ignore_cls(labels_subclips_curr, self.ignore_cls)

#             # # convert labels_subclips to one-hot format
#             # labels_subclips_out = {}
#             # labels_subclips_ignore_index = {}
#             # for key, val in labels_subclips.items():
#             #     val_tmp = val.clone()
#             #     # we first assign those ignore classes 0, so that the code works
#             #     # the runner will filter out these ignore classes later
#             #     subclips_ignore_index = val == self.ignore_cls
#             #     val_tmp[subclips_ignore_index] = 0
#             #     labels_subclips_ignore_index[key] = subclips_ignore_index

#             #     try:
#             #         val_one_hot = convert_to_one_hot(val_tmp, self.num_classes[key], self.label_smoothing[key])
#             #     except:
#             #         val_one_hot = convert_to_one_hot(val_tmp, self.num_classes[key], self.label_smoothing)

#             #     labels_subclips_out[key] = val_one_hot

#             # if batch_wo_ignore_index.sum() <= 1:
#             #     # we don't do mixup here, since there is only one single batch wo ignore index
#             #     # print(labels_subclips_out["action"].shape)
#             #     return x_video, labels_out, labels_subclips_out, labels_subclips_ignore_index

#         mixup_lambda = self.mixup_beta_sampler.sample()

#         # mix inputs
#         try:
#             x_out = {
#                 modk: _mix(x.clone(), batch_wo_ignore_index, mixup_lambda)
#                 for modk, x in x_video.items()
#             }
#         except:
#             x_out = _mix(x_video.clone(), batch_wo_ignore_index, mixup_lambda)

#         # mix labels
#         labels_out = {
#             key: _mix(val, batch_wo_ignore_index, mixup_lambda)
#             for key, val in labels_out.items()
#         }

#         if labels_subclips is None:
#             return x_video, labels_out, None, None

#         # mix labels of past frames
#         labels_subclips_out = {
#             key: _mix(val, batch_wo_ignore_index, mixup_lambda)
#             for key, val in labels_subclips_out.items()
#         }

#         return x_out, labels_out, labels_subclips_out, None



# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.
"""
Implementation of mixup and cutmix augmentation techinques based on :
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899) #
Code Reference: 
    CutMix: https://github.com/clovaai/CutMix-PyTorch

"""

import torch
import numpy as np

def convert_to_one_hot(targets, num_classes, on_value=1.0, off_value=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        on_value (float): Target Value for ground truth class.
        off_value (float): Target Value for other classes.This value is used for
            label smoothing.
    Return:
        One hot encoding of targets
    Raises:
        None
    """

    targets = targets.long().view(-1, 1) # (B) - > (B, 1)
    return torch.full(
        (targets.size()[0], num_classes), off_value, device=targets.device
    ).scatter_(1, targets, on_value) # (B, 1) -> (B, num_classes) = one hot encoded vector
 


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes and the mixed classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixup/cutmix. Deciides the amount of mixup (probability)
        smoothing (float): Label smoothing value. (allows higher error margins)
    Returns:
        One-hot encoding of target variable containing probabilities of the mixed classes
    """
    try:
        off_value = smoothing / num_classes
    except:
        off_value = smoothing / num_classes['action']
        num_classes =  num_classes['action']

    on_value = 1.0 - smoothing + off_value

    target1 = convert_to_one_hot(
        target,
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )

    target2 = convert_to_one_hot(
        target.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )

    return target1 * lam + target2 * (1.0 - lam)


def rand_bbox(img_shape, lam, margin=0.0, count=None):
    """
    Generates a random square bbox based on lambda value.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin
            (reduce amount of box outside image)
        count (int): Number of bbox to generate
    Return:
        Bounding box coordinates
    Raises:
        None
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def get_cutmix_bbox(img_shape, lam, correct_lam=True, count=None):
    """
    Generates the box coordinates for cutmix.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        correct_lam (bool): Apply lambda correction when cutmix bbox clipped by
            image borders.
        count (int): Number of bbox to generate
    """

    yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class MixUp:
    """
    Apply mixup and/or cutmix for images/videos at batch level.
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
        Features (https://arxiv.org/abs/1905.04899)
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        mix_prob=1.0,
        switch_prob=0.5,
        correct_lam=True,
        label_smoothing=0.1,
        num_classes=1000,
    ):
        """
        Args:
            mixup_alpha (float): Mixup alpha value.
            cutmix_alpha (float): Cutmix alpha value.
            mix_prob (float): Probability of applying mixup or cutmix.
            switch_prob (float): Probability of switching to cutmix instead of
                mixup when both are active.
            correct_lam (bool): Apply lambda correction when cutmix bbox
                clipped by image borders.
            label_smoothing (float): Apply label smoothing to the mixed target
                tensor. If label_smoothing is not used, set it to 0.
            num_classes (int): Number of classes for target.
        """
        self.mixup_alpha = mixup_alpha #lambda
        self.cutmix_alpha = cutmix_alpha #cutmix lambda
        self.mix_prob = mix_prob #mixup merging probabbility [0-1]
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.correct_lam = correct_lam

    def _get_mixup_params(self):
        lam = 1.0
        use_cutmix = False
        if np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        lam, use_cutmix = self._get_mixup_params()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = get_cutmix_bbox(
                x.shape,
                lam,
                correct_lam=self.correct_lam,
            )
            x[..., yl:yh, xl:xh] = x.flip(0)[..., yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped) #inplace operations, no need to return x
        return lam

    def __call__(self, x, target, past_target=None):
        # assert len(x) > 1, "Batch size should be greater than 1 for mixup."
        lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)

        if past_target is not None:
            past_target = mixup_target(past_target, self.num_classes, lam, self.label_smoothing)
            return x, target, past_target

        return x, target
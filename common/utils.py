# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import print_function
from typing import List, Dict

import errno
import os
from pathlib import Path
import logging
import submitit
import cv2

import torch
import torch.distributed as dist


def topk_correct(preds:torch.tensor, labels:torch.tensor, ks:list):
    """
    Note: This algorithm works only for classification problems whose labels are int in range [0 : (num_classes-1)]
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """

    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"

    #find top k predictions for each sample
    _top_k_vals, _top_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )

    _top_k_inds = _top_k_inds.t() #reshape to (top_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(_top_k_inds)
    # print(f"top_k_inds: {_top_k_inds}  labels: {labels}")

    #Indeces are equall to labels i.e. classes
    top_k_correct = _top_k_inds.eq(rep_max_k_labels)
    num_correct = [top_k_correct[:k, :].float().sum() for k in ks]

    return num_correct


def topk_accuracies(preds, labels, ks):
    """
    Computes top-k accuracies
    Note: This algorithm works only for classification problems whose labels are int in range [0 : (num_classes-1)]
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        Accuracies for each k
    """

    num_topk_correct = topk_correct(preds=preds, labels=labels, ks=ks)
    return [(x/preds.size(0)) * 100.0 for x in num_topk_correct]


def accuracy(output, target, topk=(1, ), mixup_enabled:bool=False):
    """Computes the accuracy over the k top predictions
    for the specified values of k
    Args:
        output (*, K) predictions
        target (*, ) targets
    """
    if torch.all(target < 0):
        return [
            torch.zeros([], device=output.device) for _ in range(len(topk))
        ]
    with torch.no_grad():
        # flatten the initial dimensions, to deal with 3D+ input
        # print(f" Output shape: {output.shape} Labels shape: {target.shape}")


        if mixup_enabled:
            #Uniting the probabilities of top2 predictions (soft labels) into 1 to calculate accuracy
            _top_max_k_vals, top_max_k_inds = torch.topk(
                target, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(target.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(target.shape[0]), top_max_k_inds[:, 1]
            pred = output.detach()
            pred[idx_top1] += pred[idx_top2]
            pred[idx_top2] = 0.0
            target = top_max_k_inds[:, 0]

            num_top_k_correct = topk_correct(pred, target, topk)
            res = []
            for k in range(len(num_top_k_correct)):
                res.append(num_top_k_correct[k]* (100/target.shape[0]))
        
        else:

        #     print(output.shape)
        #     print(output)

        #     print(target.shape)

        #     output = output.flatten(0, -2)
        #     target = target.flatten()

        # # Now compute the accuracy
        #     maxk = max(topk)
        #     batch_size = target.size(0)

        #     _, pred = output.topk(maxk, 1, True, True)
        #     pred = pred.t()
        #     correct = pred.eq(target[None])
        #     print(f"Prediction: {pred}  Target: {target}")

        #     res = []
        #     for k in topk:
        #         correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        #         res.append(correct_k * (100.0 / batch_size))
            res = topk_accuracies(output, target, topk) 
        return res




def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master, logger):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    if not is_master:
        # Don't print anything except FATAL
        logger.setLevel(logging.ERROR)
        logging.basicConfig(level=logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(logger, dist_backend='nccl'):
    dist_info = dict(
        distributed=False,
        rank=0,
        world_size=1,
        gpu=0,
        dist_backend=dist_backend,
        dist_url=get_init_file(None).as_uri(),
    )
    # If launched using submitit, get the job_env and set using those
    try:
        job_env = submitit.JobEnvironment()
    except RuntimeError:
        job_env = None
    if job_env is not None:
        dist_info['rank'] = job_env.global_rank
        dist_info['world_size'] = job_env.num_tasks
        dist_info['gpu'] = job_env.local_rank
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist_info['rank'] = int(os.environ["RANK"])
        dist_info['world_size'] = int(os.environ['WORLD_SIZE'])
        dist_info['gpu'] = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        dist_info['rank'] = int(os.environ['SLURM_PROCID'])
        dist_info['gpu'] = dist_info['rank'] % torch.cuda.device_count()
    elif 'rank' in dist_info:
        pass
    else:
        print('Not using distributed mode')
        dist_info['distributed'] = False
        return dist_info

    dist_info['distributed'] = True

    torch.cuda.set_device(dist_info['gpu'])
    dist_info['dist_backend'] = dist_backend
    print('| distributed init (rank {}): {}'.format(dist_info['rank'],
                                                    dist_info['dist_url']),
          flush=True)
    torch.distributed.init_process_group(backend=dist_info['dist_backend'],
                                         init_method=dist_info['dist_url'],
                                         world_size=dist_info['world_size'],
                                         rank=dist_info['rank'])
    setup_for_distributed(dist_info['rank'] == 0, logger)
    return dist_info


def get_shared_folder(name) -> Path:
    # Since using hydra, which figures the out folder
    return Path('./').absolute()


def get_init_file(name):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(name)), exist_ok=True)
    init_file = get_shared_folder(name) / 'sync_file_init'
    return init_file


def gather_tensors_from_all(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_dist_avail_and_initialized():
        gathered_tensors = [
            torch.zeros_like(tensor)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
    else:
        gathered_tensors = [tensor]

    return gathered_tensors


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    gathered_tensors = gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def get_video_info(video_path: Path, props: List[str]) -> Dict[str, float]:
    """
    Given the video, return the properties asked for
    """
    output = {}
    cam = cv2.VideoCapture(str(video_path))
    if 'fps' in props:
        output['fps'] = cam.get(cv2.CAP_PROP_FPS)
        # print(output['fps'])
    if 'len' in props:
        fps = cam.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            output['len'] = 0
        else:
            output['len'] = (cam.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    cam.release()
    return output

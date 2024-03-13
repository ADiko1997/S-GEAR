
"""Cross entropy loss, that works with multi-dim input."""
import torch
import torch.nn as nn
from common.cluster import KmeansAssigner
from typing import Dict, Sequence, Union



class MultiDimCrossEntropy(nn.CrossEntropyLoss):
    """Will reshape the flatten initial dimensions and then incur loss"""

    def forward(self, inp, tgt,
                one_hot: bool = False,
                ignore_index: Union[torch.Tensor, None] = None):
        """
        Args:
            inp: (*, C)
            tgt: (*, )
            one_hot: whether the labels are already one-hotted
            ignore_index: index of inputs to be ignored
        """
        
        # if one_hot:
        #     if inp.ndim == tgt.ndim + 1: #In case of fine graned labeling
        #         inp.reshape_(-1, inp.shape[-1])
        #     assert inp.ndim == tgt.ndim

        # else:
        #     assert inp.ndim == tgt.ndim + 1
        #     assert inp.shape[:-1] == tgt.shape

        inp = inp.reshape(-1, inp.size(-1))
        tgt = tgt.reshape(-1,) if not one_hot else tgt.reshape(-1, tgt.size(-1))

        # if ignore_index is not None:
        #     assert one_hot, "Target should be one-hotted."
        #     ignore_index = ignore_index.reshape(-1,)
        #     keep_index = ~ignore_index
        #     inp = inp[keep_index]
        #     tgt = tgt[keep_index]

        res = super().forward(inp, tgt)

        return res



class QuantizeAndCrossEntropy(MultiDimCrossEntropy):
    """Given a set of cluster centers, project the features to that before
    incurring the loss."""
    def __init__(self, centroids_fpath, norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigner = KmeansAssigner(centroids_fpath)
        self.norm = norm

    def forward(self, inp, tgt):
        """
        Args:
            inp: (*, C)
            tgt: (*, C)
            Will reshape the flatten initial dimensions and then incur loss
        """
        # Normalize L2 both target and input, since that's how I'm computing
        # centroids
        if self.norm:
            inp = nn.functional.normalize(inp, dim=-1, p=2)
            tgt = nn.functional.normalize(tgt, dim=-1, p=2)
        # assign the GT and predictions to the centroids
        inp_proj = torch.mm(inp.flatten(0, 1),
                            self.centroids.t()).view(inp.shape[:-1] +
                                                     self.centroids.shape[:1])
        # the weights of project layer are the centroids, so pick from there
        tgt_proj_q = self.assigner(tgt)
        return super().forward(inp_proj, tgt_proj_q)

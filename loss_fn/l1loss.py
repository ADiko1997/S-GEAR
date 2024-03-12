"""Normalized L1 loss """

import torch.nn as nn

class NormalizedL1(nn.L1Loss):
    def forward(self, inp, target, *args, **kwargs):
        """
        Input:
            inp (torch.tensor): Output of the network in embeddings
            target (torch.tensor): Target embeddings
        """

        inp = nn.functional.normalize(inp, dim=-1, p=1)
        target = nn.functional.normalize(target, dim=-1, p=1)
        return super().forward(inp, target, *args, **kwargs)
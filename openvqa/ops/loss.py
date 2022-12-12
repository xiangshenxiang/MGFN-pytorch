import torch.nn as nn
import torch
from torch.autograd import Variable

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, imgs, caps):
        scores = torch.mm(imgs, caps.t())
        diag = scores.diag()

        cost_s = torch.clamp((self.margin - diag).expand_as(scores) + scores, min=0)

        # compare every diagonal score to scores in its row (i.e, all
        # contrastive sentences for each image)
        cost_im = torch.clamp((self.margin - diag.view(-1, 1)).expand_as(scores) + scores, min=0)
        # clear diagonals
        diag_s = torch.diag(cost_s.diag())
        diag_im = torch.diag(cost_im.diag())

        cost_s = cost_s - diag_s
        cost_im = cost_im - diag_im

        return cost_s.sum() + cost_im.sum()
##lossT
class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax

    def forward(self, imgs, caps):
        scores = torch.matmul(imgs, caps.t())
        diag = scores.diag()

        # Reducing the score on diagonal so there are not selected as hard negative
        scores = (scores - 2 * torch.diag(scores.diag()))

        sorted_cap, _ = torch.sort(scores, 0, descending=True)
        sorted_img, _ = torch.sort(scores, 1, descending=True)

        # Selecting the nmax hardest negative examples
        max_c = sorted_cap[:self.nmax, :]
        max_i = sorted_img[:, :self.nmax]

        # Margin based loss with hard negative instead of random negative
        neg_cap = torch.sum(torch.clamp(max_c + (self.margin - diag).view(1, -1).expand_as(max_c), min=0))
        neg_img = torch.sum(torch.clamp(max_i + (self.margin - diag).view(-1, 1).expand_as(max_i), min=0))
        loss = neg_cap + neg_img

        return loss

##重要 lossD
class DiversityLoss(object):
    def __init__(self, args):
        super(DiversityLoss, self).__init__()
        self.attention_hop = args.attn_hop
        self.coeff=args.penalty_coeff

        self.I = Variable(torch.zeros(args.batch_size, self.attention_hop, self.attention_hop)).cuda()
        for p in range(args.batch_size):
            for q in range(self.attention_hop):
                self.I.data[p][q][q] = 1

    def cal_loss(self, attention_map):
        ##装置
        attention_map_T = torch.transpose(attention_map, 1, 2).contiguous()
        diversity_loss = self.Frobenius(torch.bmm(attention_map, attention_map_T) - self.I[:attention_map.size(0)])

        return self.coeff * diversity_loss

    def Frobenius(self, mat):
        size = mat.size()
        if len(size) == 3:  # batched matrix
            ret = (torch.sum(torch.sum((mat ** 2), 1), 1) + 1e-10) ** 0.5
            return torch.sum(ret) / size[0]
        else:
            raise Exception('matrix for computing Frobenius norm should be with 3 dims')


# class My_loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x, y):
#         return torch.mean(torch.pow((x - y), 2))

import warnings

# from .distance import PairwiseDistance
from torch.nn.modules import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

from torch import Tensor
from typing import Callable, Optional
class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
class My_loss(_Loss):

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None) -> None:
        super(My_loss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.weight: Optional[Tensor]
        self.pos_weight: Optional[Tensor]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)
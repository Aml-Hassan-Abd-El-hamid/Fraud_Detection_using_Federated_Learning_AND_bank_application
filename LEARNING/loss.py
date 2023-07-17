import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Focal_Loss(nn.Module):
    """
    source for that function: https://github.com/mathiaszinnen/focal_loss_torch
    Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2

    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(target.shape[0])
        weights = target * self.weights
        return weights.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
            ) -> Tensor:
        
        #convert all ignore_index elements to zero to avoid error in one_hot
        #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        target = target * (target!=self.ignore_index) 
        target = target.view(-1)
        
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            #print("damnnn")
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        #print(x.dim())
        target=target.to(torch.int64)
        mask = target == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        #print("num_classes",num_classes)
        target = self._process_target(target, num_classes, mask)
        weights = self._get_weights(target).to(x.device)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        #print(self._reduce(loss, mask, weights))
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x

def roc_star_loss( _y_true, y_pred, gamma, _epoch_true, epoch_pred):
    """
    source for that function: https://github.com/iridiumblue/roc-star
    Nearly direct loss function for AUC.
    See article,
    C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
    https://github.com/iridiumblue/articles/blob/master/roc_star.md
        _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        gamma  : `Float` Gamma, as derived from last epoch.
        _epoch_true: `Tensor`.  Targets (labels) from last epoch.
        epoch_pred : `Tensor`.  Predicions from last epoch.
    """
    #print(" _y_true, y_pred", _y_true.shape, y_pred.shape)
    #convert labels to boolean
    y_true = (_y_true>=0.50)
    epoch_true = (_epoch_true>=0.50)

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred)*1e-8

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    epoch_pos = epoch_pred[epoch_true]
    epoch_neg = epoch_pred[~epoch_true]

    # Take random subsamples of the training set, both positive and negative.
    max_pos = 1000 # Max number of positive training samples
    max_neg = 1000 # Max number of positive training samples
    cap_pos = epoch_pos.shape[0]
    cap_neg = epoch_neg.shape[0]
    epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
    epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    # sum positive batch elements agaionst (subsampled) negative elements
    if ln_pos>0 :
        pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
        neg_expand = epoch_neg.repeat(ln_pos)

        diff2 = neg_expand - pos_expand + gamma
        l2 = diff2[diff2>0]
        m2 = l2 * l2
        len2 = l2.shape[0]
    else:
        m2 = torch.tensor([0], dtype=torch.float).to(device)
        len2 = 0

    # Similarly, compare negative batch elements against (subsampled) positive elements
    if ln_neg>0 :
        pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
        neg_expand = neg.repeat(epoch_pos.shape[0])

        diff3 = neg_expand - pos_expand + gamma
        l3 = diff3[diff3>0]
        m3 = l3*l3
        len3 = l3.shape[0]
    else:
        m3 = torch.tensor([0], dtype=torch.float).to(device)
        len3=0

    if (torch.sum(m2)+torch.sum(m3))!=0 :
       res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
       #code.interact(local=dict(globals(), **locals()))
    else:
       res2 = torch.sum(m2)+torch.sum(m3)

    res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

    return res2


def epoch_update_gamma(y_true,y_pred, epoch=-1,delta=2,device="cpu"):
    """
    source for that function: https://github.com/iridiumblue/roc-star
    Calculate gamma from last epoch's targets and predictions.
    Gamma is updated at the end of each epoch.
    y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
    y_pred: `Tensor` . Predictions.
    """
    DELTA = delta
    SUB_SAMPLE_SIZE = 2000.0
    pos = y_pred[y_true==1]
    neg = y_pred[y_true==0] # yo pytorch, no boolean tensors or operators?  Wassap?
    # subsample the training set for performance
    cap_pos = pos.shape[0]
    cap_neg = neg.shape[0]
    pos = pos[torch.rand_like(pos) < SUB_SAMPLE_SIZE/cap_pos]
    neg = neg[torch.rand_like(neg) < SUB_SAMPLE_SIZE/cap_neg]
    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]
    pos_expand = pos.view(-1,1).expand(-1,ln_neg).reshape(-1)
    neg_expand = neg.repeat(ln_pos)
    diff = neg_expand - pos_expand
    ln_All = diff.shape[0]
    Lp = diff[diff>0] # because we're taking positive diffs, we got pos and neg flipped.
    ln_Lp = Lp.shape[0]-1
    diff_neg = -1.0 * diff[diff<0]
    diff_neg = diff_neg.sort()[0]
    ln_neg = diff_neg.shape[0]-1
    ln_neg = max([ln_neg, 0])
    left_wing = int(ln_Lp*DELTA)
    left_wing = max([0,left_wing])
    left_wing = min([ln_neg,left_wing])
    default_gamma=torch.tensor(0.2, dtype=torch.float).to(device)
    if diff_neg.shape[0] > 0 :
       gamma = diff_neg[left_wing]
    else:
       gamma = default_gamma # default=torch.tensor(0.2, dtype=torch.float).cuda() #zoink
    L1 = diff[diff>-1.0*gamma]
    ln_L1 = L1.shape[0]
    if epoch > -1 :
        return gamma
    else :
        return default_gamma
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LDAMLoss(nn.Module):
    
    def __init__(self, device, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()

        self.device = device
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # m_list = torch.cuda.FloatTensor(m_list) torch.from_numpy
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list.to(device)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8).to(self.device)
        target=target.to(torch.int64)
        #print(target,target.shape,target.ndim,target.dtype)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        # index_float = index.type(torch.cuda.FloatTensor)
        index_float = index.type(torch.FloatTensor).to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        # print(index_float)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
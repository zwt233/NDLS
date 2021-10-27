import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class DNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DNN, self).__init__()
        self.fcn1 = nn.Linear(nfeat, nhid)
        self.fcn2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fcn1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcn2(x)
        return F.log_softmax(x, dim=1)
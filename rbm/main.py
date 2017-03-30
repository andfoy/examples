import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class RBM(nn.Module):
    def __init__(self, D_in, H, D_out, N):
        super(RBM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.visible_units = nn.Linear(D_in, H)
        self.hidden_units = nn.Linear(H, d_out)
        self.hidden_units.weight = self.visible_units.weight.transpose()
        self.N = N
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.visible_units.weight.data.uniform_(-initrange, initrange)


class RBMLayer(autograd.Function):
    def __init__(self, D_in, H, N):
        autograd.Function.__init__(self)
        self.input_size = D_in
        self.hidden_size = H
        self.cd = N
        self.weight = Parameter(torch.Tensor(self.hidden_size,
                                             self.input_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def binarize(self, x):
        rand_source = torch.Tensor(x.size()).uniform_(0, 1)
        return 0 + x > rand_source

    def vis_to_hid(self, x):
        return x.mm(self.weight)

    def forward(self, input):
        vis_probs = self.binarize(input)
        return self.vis_to_hid(vis_probs)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class RBM(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(RBM, self).__init__()
        self.visible_units = nn.Sigmoid(nn.Linear(D_in, H))
        self.hidden_units = nn.Sigmoid(nn.Linear(H, d_out))

    def init_weights(self):
        initrange = 0.1
        self.visible_units.weight.data.uniform_(-initrange, initrange)
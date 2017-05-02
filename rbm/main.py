from __future__ import print_function

import torch
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait'
                         'before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class RBM(nn.Module):
    def __init__(self, D_in, H, N):
        super(RBM, self).__init__()
        self.in_dim = D_in
        self.N = N
        self.rbm = RBMLayer(D_in, H, N)
        self.myparameters = Parameter(self.rbm.weight)

    def forward(self, x):
        return self.rbm(x.view(-1, self.in_dim).t())


class RBMLayer(autograd.Function):
    def __init__(self, D_in, H, N):
        autograd.Function.__init__(self)
        self.input_size = D_in
        self.hidden_size = H
        self.cd = N
        # self.sigmoid = nn.Sigmoid()
        self.weight = torch.Tensor(self.hidden_size,
                                   self.input_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-stdv, stdv)

    def configuration_grad(self, vis_state, hid_state):
        # print(hid_state.t().size(1))
        grad = vis_state.mm(hid_state.t()) / vis_state.size(1)
        return grad

    def binarize(self, x):
        # print(x)
        rand_source = Variable(torch.Tensor(x.size()).uniform_(0, 1))
        tmp_x = x
        if not isinstance(x, Variable):
            tmp_x = Variable(x)
        bin_val = tmp_x > rand_source
        return torch.FloatTensor(x.size()).copy_(bin_val.data)

    def vis_to_hid(self, x):
        return torch.sigmoid(self.weight.mm(x))

    def hid_to_vis(self, x):
        return torch.sigmoid(self.weight.t().mm(x))

    def forward(self, input):
        vis_bin = self.binarize(input)
        hid_probs = self.vis_to_hid(vis_bin)
        self.save_for_backward(hid_probs)
        return vis_bin, hid_probs

    def backward(self, vis_bin, hid_probs):
        # hid_probs = self.saved_tensors
        hid_bin = self.binarize(hid_probs)
        # print(hid_bin)
        # print(vis_bin)
        grad1 = self.configuration_grad(vis_bin.data, hid_bin)
        # print("Grad1")
        # print(grad1)
        for i in range(0, self.cd):
            vis_probs = self.hid_to_vis(hid_bin)
            vis_bin = self.binarize(vis_probs)
            hid_probs = self.vis_to_hid(vis_bin)
            if i == self.cd - 1:
                hid_bin = hid_probs
                break
            hid_bin = self.binarize(hid_probs)
        grad2 = self.configuration_grad(vis_bin, hid_bin)
        # print(grad1)
        # print("Grad2")
        # print(grad2)
        return grad1 - grad2


model = RBM(784, 500, 1)
if args.cuda:
    model.cuda()

# optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    grad = None
    momentum = torch.zeros(500, 784)
    velocity = torch.zeros(500, 784)
    alpha = 1e-3
    eps = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        # data = data / torch.max(data)
        if args.cuda:
            data = data.cuda()
        # optimizer.zero_grad()
        vis, hid = model(data)
        if not torch.is_tensor(hid):
            hid = hid.data
        # loss = loss_function(out)
        # loss.backward()
        grad = model.rbm.backward(vis, hid)
        momentum = beta1 * momentum + (1 - beta1) * grad
        velocity = beta2 * velocity + (1 - beta2) * grad**2
        # print(momentum)
        # print(velocity)
        model.rbm.weight += alpha * momentum / (torch.sqrt(velocity) + eps)
        # print(grad)
        # optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader)))
    print('====> Epoch: {}'.format(epoch))


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        torch.save(model.state_dict(), 'rbm')

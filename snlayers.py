import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

# define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)


def max_singular_value(weight, u=None, num_iter=1):
    """

    :param weight: nn.Parameter matrix
    :param u: initial estimation of the right largest singular value of weight matrix
    :param num_iter: number of iterations
    :return: (spectral_norm, new estimation of u )
    """
    if not num_iter >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.randn((1, weight.size(0)), dtype=torch.float32, device=weight.data.device, layout=weight.data.layout)
    _u = u
    for _ in range(num_iter):
        _v = _l2normalize(torch.matmul(_u, weight.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(weight.data, 0, 1)), eps=1e-12)
    spectral_norm = torch.sum(F.linear(_u, torch.transpose(weight.data, 0, 1)) * _v)
    return spectral_norm, _u


class SNLinear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Attributes:
           W (Tensor): Spectral normalized weight
           u (Tensor): the right largest singular value of W.
       """

    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        spec_norm, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / spec_norm

    def forward(self, x):
        return F.linear(x, self.W_, self.bias)


class SNConv2d(nn.modules.conv._ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.
    Attributes:
        W(Tensor): Spectrally normalized weight
        u (Tensor): the right largest singular value of W.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        spec_norm, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / spec_norm

    def forward(self, x):
        return F.conv2d(x, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

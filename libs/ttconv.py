import torch
from torch import nn
from torch.nn.functional import conv2d, relu
from numpy import prod, power
from string import ascii_lowercase as letters
from typing import Iterable
import opt_einsum as oe


class TTConv(nn.Module):
    def __init__(self, in_c_modes, out_c_modes, kernel_size, ranks,
                 stride=1, padding=0, initialize=nn.init.xavier_normal_):
        super().__init__()
        self.in_c_modes = in_c_modes
        self.out_c_modes = out_c_modes
        self.ranks = [*ranks, 1]

        self.core0 = nn.Conv2d(1, self.ranks[0], kernel_size, stride=stride, padding=padding)
        initialize(self.core0.weight)
        self.cores = []
        for i in range(len(self.in_c_modes)):
            self.cores.append(torch.empty(self.out_c_modes[i] * self.ranks[i+1], self.ranks[i] * self.in_c_modes[i]))
            initialize(self.cores[-1])

    def forward(self, X: torch.Tensor):
        in_c, h, w = X.shape[-3:]
        tmp = X.reshape((-1, 1, h, w)) # (batch_size * in_c, 1, h, w)

        tmp = self.core0(tmp) # (batch_size * in_c, ranks[0], h, w)

        h, w = tmp.shape[-2:]
        tmp = tmp.reshape((-1, in_c, self.ranks[0], h, w)) # (batch_size, in_c, ranks[0], h, w)
        tmp = tmp.transpose(0, 2) # (ranks[0], in_c, batch_size, h, w)

        for i in range(len(self.in_c_modes)):
            tmp = tmp.reshape((self.ranks[i] * self.in_c_modes[i], -1))
            # (ranks[i] * in_c_modes[i], prod(in_c_modes[i+1:]) * batch_size * h * w, out_c_modes[0], ..., out_c_modes[i-1])
            tmp = self.cores[i].matmul(tmp)
            # (out_c_modes[i] * ranks[i+1], prod(in_c_modes[i+1:]) * batch_size * h * w, out_c_modes[0], ..., out_c_modes[i-1])
            tmp = tmp.reshape((self.out_c_modes[i], -1))
            # (out_c_modes[i], ranks[i+1] * prod(in_c_modes[i+1:]) * batch_size * h * w, out_c_modes[0], ..., out_c_modes[i-1])
            tmp = tmp.transpose(0, -1)
            # (ranks[i + 1] * prod(in_c_modes[i+1:]) * batch_size * h * w, out_c_modes[0], ..., out_c_modes[i-1], out_c_modes[i])

        out_c = prod(self.out_c_modes)
        out = tmp.reshape((-1, h, w, out_c)).permute((0, 3, 1, 2))

        return out


def _hrinchuk_init(core, in_c_modes, ranks, kernel_size):
    fan_in = prod(in_c_modes) * kernel_size ** 2
    old_var = 2 / fan_in
    new_std_base = old_var / prod(ranks)
    new_std_power = 0.5 / (len(ranks) + 1)
    nn.init.normal_(core, 0, power(new_std_base, new_std_power))


class TTConvEinsum(nn.Module):
    def __init__(self, in_c_modes, out_c_modes, kernel_size, ranks,
                 stride=1, padding=0, init=None, mask=None):
        super().__init__()

        self.in_c_modes = list(in_c_modes)
        self.out_c_modes = list(out_c_modes)
        self.d = len(self.in_c_modes)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init = init

        if self.init is None:
            self.init = lambda core: _hrinchuk_init(
                core, in_c_modes, self.ranks, self.kernel_size)

        if not isinstance(ranks, Iterable):
            ranks = [ranks] * self.d
        
        max_ranks = []
        cnt1 = kernel_size ** 2
        cnt2 = prod(in_c_modes) * prod(out_c_modes)
        for i in range(self.d):
            max_ranks.append(int(min(cnt1, cnt2)))
            factor = in_c_modes[i] * out_c_modes[i]
            cnt1 *= factor; cnt2 /= factor
        
        self.ranks = []
        for i in range(self.d):
            if ranks[i] is None:
                self.ranks.append(max_ranks[i])
            elif isinstance(ranks[i], int):
                self.ranks.append(ranks[i])
            elif isinstance(ranks[i], float):
                self.ranks.append(int(ranks[i] * max_ranks[i]))
        self.ranks.append(1)

        self.core0 = nn.Parameter(
            torch.empty(self.ranks[0], 1, self.kernel_size, self.kernel_size)
        )
        self.init(self.core0)
        if mask is not None:
            with torch.no_grad():
                self.core0 *= mask
        
        self.cores = []
        for i in range(self.d):
            self.cores.append(
                nn.Parameter(
                    torch.empty(self.in_c_modes[i], self.out_c_modes[i],
                                self.ranks[i], self.ranks[i+1])
                )
            )
            self.init(self.cores[-1])
        self.cores = nn.ParameterList(self.cores)
    
    def _apply_convolution(self, X: torch.Tensor):
        h, w = X.shape[-2:]
        tmp = X.reshape((-1, 1, h, w)) # (batch_size * in_c, 1, h, w)

        tmp = conv2d(tmp, self.core0, stride=self.stride, padding=self.padding)

        return tmp

    def forward(self, X: torch.Tensor):
        tmp = self._apply_convolution(X) 
        # (batch_size * in_c, ranks[0], h, w), possibly with new h & w

        h, w = tmp.shape[-2:]
        tmp = tmp.reshape((-1, *self.in_c_modes, self.ranks[0], h, w)) # (batch_size, in_c_modes, ranks[0], h, w)

        for i in range(self.d):
            einstr = 'B' + letters[:i] + 'I' + letters[i:self.d-1] + 'SHW,IJST->' + \
                     'B' + letters[:i] + 'J' + letters[i:self.d-1] + 'THW'
            tmp = oe.contract(einstr, tmp, self.cores[i], optimize='optimal')
            # (batch_size, in_c_modes[:i+1], out_c_modes[i+1:], ranks[i+1], h, w)

        out_c = prod(self.out_c_modes)
        out = tmp.reshape((-1, out_c, h, w))

        return out
    
    def _precompute_filter(self):
        full = self.core0.squeeze(1)
        for i in range(self.d):
            einstr = letters[:2*i] + 'SHW,IJST->' + letters[:i] + 'J' + letters[i:2*i] + 'ITHW'
            full = oe.contract(einstr, full, self.cores[i], optimize='optimal')
        full = full.reshape(prod(self.out_c_modes), prod(self.in_c_modes), self.kernel_size, self.kernel_size)

        return full
    
    def _forward_precompute(self, X: torch.Tensor):
        h, w = X.shape[-2:]
        tmp = X.reshape((-1, prod(self.in_c_modes), h, w))

        full = self._precompute_filter()
        out = conv2d(tmp, full, stride=self.stride, padding=self.padding)

        return out


class TTConvEinsumContract(TTConvEinsum):
    def __init__(self, in_c_modes, out_c_modes, kernel_size, ranks=None,
                 stride=1, padding=0, batch_size=128, input_size=32, init=None, mask=None):
        super().__init__(in_c_modes, out_c_modes, kernel_size, ranks,
                         stride, padding, init, mask)
        self.batch_size = batch_size
        self.input_size = input_size
        self.after_conv_size = int((input_size + 2 * padding - kernel_size) / stride + 1)

        self.ein_funcs = []
        for i in range(self.d):
            einstr = 'B' + letters[:i] + 'I' + letters[i:self.d-1] + 'SHW,IJST->' + \
                     'B' + letters[:i] + 'J' + letters[i:self.d-1] + 'THW'
            shape1 = tuple([self.batch_size] + self.in_c_modes[:i+1] + self.out_c_modes[i+1:] +
                           [self.ranks[i], self.after_conv_size, self.after_conv_size])
            shape2 = self.cores[i].shape
            self.ein_funcs.append(oe.contract_expression(einstr, shape1, shape2))

    def forward(self, X: torch.Tensor):
        tmp = self._apply_convolution(X) 
        # (batch_size * in_c, ranks[0], h, w), possibly with new h & w

        h, w = tmp.shape[-2:]
        tmp = tmp.reshape((-1, *self.in_c_modes, self.ranks[0], h, w)) # (batch_size, in_c_modes, ranks[0], h, w)

        d = len(self.in_c_modes)
        for i in range(d):
            tmp = self.ein_funcs[i](tmp, self.cores[i])
            # (batch_size, in_c_modes[:i+1], out_c_modes[i+1:], ranks[i+1], h, w)

        out_c = prod(self.out_c_modes)
        out = tmp.reshape((-1, out_c, h, w))

        return out


class TTConvGaussian(TTConvEinsumContract):
    def __init__(self, in_c_modes, out_c_modes, kernel_size, ranks=None,
                 stride=1, padding=0, batch_size=128, input_size=32,
                 init=None, mask=None, sigma=0.3, sigma_lower_bound=0.0):
        super().__init__(in_c_modes, out_c_modes, kernel_size, ranks,
                         stride, padding, batch_size, input_size, init, mask)

        self.sigma_lower_bound = sigma_lower_bound
        self.sigma = nn.Parameter(torch.tensor(max(0, sigma - sigma_lower_bound)))

    def _apply_convolution(self, X: torch.Tensor):
        std = self.sigma_lower_bound + relu(self.sigma)
        grid = torch.arange(-self.kernel_size//2+1, self.kernel_size//2+1) ** 2
        grid = grid.to(self.sigma.device)
        power = -(grid.unsqueeze(0) + grid.unsqueeze(1)) / (2 * std ** 2)
        mask = torch.exp(power) / (2 * torch.pi * std)

        h, w = X.shape[-2:]
        tmp = X.reshape((-1, 1, h, w)) # (batch_size * in_c, 1, h, w)

        tmp = conv2d(tmp, mask * self.core0, stride=self.stride, padding=self.padding)

        return tmp
    
    def obtain_mask(self):
        std = self.sigma_lower_bound + relu(self.sigma)
        grid = torch.arange(-self.kernel_size//2+1, self.kernel_size//2+1) ** 2
        grid = grid.to(std.device)
        power = -(grid.unsqueeze(0) + grid.unsqueeze(1)) / (2 * std ** 2)
        mask = torch.exp(power) / (2 * torch.pi * std)
        
        sz = torch.count_nonzero(mask[self.kernel_size//2] > 0.001)
        lb = int((self.kernel_size-sz)/2)
        rb = int((self.kernel_size+sz)/2)
        result = mask[lb:rb, lb:rb]

        return sz.item(), result

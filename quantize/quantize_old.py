import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function

# TODO add other quantization methods


def calc_threshold(aw, method='TTQ'):
    if method == 'TTQ':
        # return 0.05 * aw.max()
        return 0.7 * aw.mean()
    elif method == 'CTQ':
        # return 0.05 * aw.data.view(aw.shape[0], -1).max(dim=1)
        # w_max, _ = aw.data.view(aw.shape[0], -1).max(dim=1)
        # return 0.1 * w_max.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return 0.7 * aw.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)


class Quantize(Function):

    @staticmethod
    def forward(ctx, fp_weight, wp, wn, method='TTQ'):
        if method == 'TTQ':
            abs_weight = fp_weight.abs()
            threshold = calc_threshold(abs_weight)
            mask_p = torch.gt(fp_weight, threshold)
            mask_n = torch.lt(fp_weight, -threshold)
            mask_z = torch.add(mask_p, mask_n)
            valid = abs_weight * mask_z.float()
            scale = valid.sum() / mask_z.sum()
            # scale = scale.view(1, 1, 1, 1)
            # wp = wp.view(1, 1, 1, 1)
            # wn = wn.view(1, 1, 1, 1)

            output = fp_weight.clone().zero_()
            # output += mask_p.float() * wp * scale
            # output += mask_n.float() * wn * scale
            output.masked_fill_(mask_p, wp[0] * scale)
            output.masked_fill_(mask_n, wn[0] * scale)

        elif method == 'CTQ':
            abs_weight = fp_weight.abs()
            threshold = calc_threshold(abs_weight, method=method)
            mask_p = torch.gt(fp_weight, threshold)
            mask_n = torch.lt(fp_weight, -threshold)
            mask_z = torch.add(mask_p, mask_n)
            valid = abs_weight * mask_z.float()
            scale = valid.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / \
                    mask_z.float().sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

            output = fp_weight.clone().zero_()
            output += mask_p.float() * wp.unsqueeze(1).unsqueeze(1).unsqueeze(1) * scale
            output += mask_n.float() * wn.unsqueeze(1).unsqueeze(1).unsqueeze(1) * scale
            # output.masked_fill_(mask_p, wp[0] * scale)
            # output.masked_fill_(mask_n, wn[0] * scale)
        else:
            raise NotImplementedError

        # ctx.save_for_backward(mask_p, mask_n, scale, method)
        ctx.mask_p = mask_p
        ctx.mask_n = mask_n
        ctx.scale = scale
        ctx.method = method
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # mask_p, mask_n, scale, method = ctx.saved_tensors
        mask_p = ctx.mask_p
        mask_n = ctx.mask_n
        scale = ctx.scale
        method = ctx.method
        if method == 'TTQ':
            grad_fp_weight = grad_output.clone()
            grad_wp = (mask_p.float() * grad_output).sum() * scale.unsqueeze(0)
            grad_wn = (mask_n.float() * grad_output).sum() * scale.unsqueeze(0)
        elif method == 'CTQ':
            grad_fp_weight = grad_output.clone()
            grad_wp = (mask_p.float() * grad_output).sum(dim=1).sum(dim=1).sum(dim=1) * scale.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
            grad_wn = (mask_n.float() * grad_output).sum(dim=1).sum(dim=1).sum(dim=1) * scale.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
        else:
            raise NotImplementedError
        return grad_fp_weight, grad_wp, grad_wn, None


def _cuda(self, device_id=None):
    super(self.__class__, self).cuda(device_id)
    # self._apply(lambda t: t.cuda(device_id))
    for param in [self._parameters[k] for k in ('wp', 'wn')]:
        param.data = param.data.cpu()
        if param._grad is not None:
            param._grad.data = param._grad.data.cpu()

    return self


def _reset_parameters(self):
    if self.method == 'TTQ':
        self.wp.data.fill_(1.)
        self.wn.data.fill_(-1.)
    elif self.method == 'CTQ':
        self.wp.data.fill_(1.)
        self.wn.data.fill_(-1.)

    super(self.__class__, self).reset_parameters()


def _str(self):
    # useful when debugging wp and wn
    """
    if self.method == 'TTQ':
        return "{}, {}".format(self.wp.data[0], self.wn.data[0])
    elif self.method == 'KMeans':
        return super(self.__class__, self).__repr__()
    """
    s = super(self.__class__, self).__repr__()[:-1]
    s += ', method={method}'
    if self.method == 'KMeans':
        s += ', nbits={nbits}'
    s += ')'
    return s.format(**self.__dict__)


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, method='KMeans', nbits=4):
        # nn.Module.__init__ has to be called at first to enable Parameter,
        # so we delay parameter reset to the end of __init__ by overriding
        # reset_parameters method with an empty object function.
        self.reset_parameters = lambda: None
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.method = method
        if self.method == 'TTQ':
            self.wp = Parameter(torch.Tensor([1.0]))
            self.wn = Parameter(torch.Tensor([-1.0]))
        elif self.method == 'KMeans':
            self.nbits = nbits
        del self.reset_parameters
        self.reset_parameters()

    def forward(self, input):
        if self.method == 'TTQ':
            q_weight = Quantize.apply(self.weight, self.wp, self.wn)
        elif self.method == 'KMeans':
            q_weight = self.weight
        return F.linear(input, q_weight, self.bias)

    reset_parameters = _reset_parameters

    # cuda = _cuda
    __repr__ = _str


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, method='TTQ', nbits=4):
                 # padding=0, dilation=1, groups=1, bias=True, method='KMeans', nbits=4):
        self.reset_parameters = lambda: None
        super(QuantConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.method = method
        if self.method == 'TTQ':
            self.wp = Parameter(torch.Tensor(1))
            self.wn = Parameter(torch.Tensor(1))
        elif self.method == 'CTQ':
            self.wp = Parameter(torch.Tensor(torch.ones_like(self.weight.mean(dim=1).mean(dim=1).mean(dim=1))))
            self.wp = Parameter(torch.Tensor(torch.ones_like(self.weight.mean(dim=1).mean(dim=1).mean(dim=1))))
        elif self.method == 'KMeans':
            self.nbits = nbits
        del self.reset_parameters
        self.reset_parameters()

    def forward(self, input):
        if self.method == 'TTQ':
            q_weight = Quantize.apply(self.weight, self.wp, self.wn, self.method)
        elif self.method == 'CTQ':
            q_weight = Quantize.apply(self.weight, self.wp, self.wn, self.method)
        elif self.method == 'KMeans':
            q_weight = self.weight
        else:
            raise NotImplementedError
        return F.conv2d(input, q_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    reset_parameters = _reset_parameters

    # cuda = _cuda
    __repr__ = _str



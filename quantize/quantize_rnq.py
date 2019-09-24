import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Function

class RNQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, aq_type, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, n_expert=16, aq_bits = -1):
        super(RNQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)
        self.n_expert = n_expert
        self.hidden = 16
        self.wp = Parameter(torch.Tensor(1))
        self.wn = Parameter(torch.Tensor(1))
        self.aq_type = aq_type
        self.aq_bits = aq_bits
        self.activation_range = nn.Parameter(torch.Tensor([6.0]))
        self._calibrate = False
        self._fix_weight = False
        self._half_wave = True
        self.routing = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        action = None
        if self.aq_type == 'rule':
            x_p = x.mean(-1).mean(-1)  # global avg pool
            mix = self.routing(x_p)
            action = self._action_wall(mix, 1, 8)
        elif self.aq_type == 'uniform':
            action = self.aq_bits
        elif self.aq_type == 'trainable':
            
        else:
            raise NotImplementedError
        inputs, weight, bias = self._quantize(inputs=x, weight=self.weight, bias=self.bias, a_bits=action)
        return F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def _quantize_weight(self, weight):
        ori_w = weight
        q_weight = Quantize.apply(self.weight, self.wp, self.wn)
        if self._fix_weight:
        # w = w.detach()
            return q_weight.detach()
        else:
            # w = ori_w + w.detach() - ori_w.detach()
            return q_weight

    def _quantize_activation(self, inputs, q_bits):
        if q_bits > 0:
            if self._calibrate:
                # threshold = self._compute_threshold(inputs.data.cpu().numpy(), self._a_bit)
                # estimate_activation_range = min(min(6.0, inputs.abs().max().item()), threshold)
                estimate_activation_range = min(6.0, inputs.abs().max().item())
                print('range:', estimate_activation_range, '  shape:', inputs.shape, '  inp_abs_max:', inputs.abs().max())
                self.activation_range.data = torch.tensor([estimate_activation_range], device=inputs.device)
                return inputs
            '''
            if self._trainable_activation_range:
                if self._half_wave:
                    ori_x = 0.5 * (inputs.abs() - (inputs - self.activation_range).abs() + self.activation_range)
                else:
                    ori_x = 0.5 * ((-inputs - self.activation_range).abs() - (inputs - self.activation_range).abs())
            else:
            '''
            if self._half_wave:
                ori_x = 0.5 * (inputs.abs() - (inputs - self.activation_range).abs() + self.activation_range)
            else:
                ori_x = 0.5 * ((-inputs - self.activation_range).abs() - (inputs - self.activation_range).abs())

            scaling_factor = self.activation_range.item() / (2. ** q_bits - 1.)
            x = ori_x.detach().clone()
            if self.aq_type =='rule':
                scaling_factor = torch.unsqueeze(torch.unsqueeze(scaling_factor, -1), -1)
            x.div_(scaling_factor).round_().mul_(scaling_factor)

            # STE
            return ori_x + x.detach() - ori_x.detach()
        else:
            return inputs

    def _action_wall(self, action, min_bit, max_bit):
        # limit the action to certain range
        action = action.float()
        lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
        action = (rbound - lbound) * action + lbound
        action = torch.round(action)
        return action  # not constrained here

    def _quantize(self, inputs, weight, bias, a_bits):
        inputs = self._quantize_activation(inputs=inputs, q_bits = a_bits)
        weight = self._quantize_weight(weight=weight)
        bias = bias
        return inputs, weight, bias

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
        #output += mask_p.float() * wp * scale
        #output += mask_n.float() * wn * scale
        output.masked_fill_(mask_p, wp[0] * scale)
        output.masked_fill_(mask_n, wn[0] * scale)

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

import torch
import torch.nn as nn
import torch.optim
import quantize as Q


def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2


def shuffle_layer(x, groups):
	batchsize, num_channels, height, width = x.data.size()
	channels_per_group = num_channels // groups
	# reshape
	x = x.view(batchsize, groups,
	           channels_per_group, height, width)
	# transpose
	x = torch.transpose(x, 1, 2).contiguous()
	# flatten
	x = x.view(batchsize, -1, height, width)
	return x


def get_split_list(in_dim, child_num):
	in_dim_list = [in_dim // child_num] * child_num
	for _i in range(in_dim % child_num):
		in_dim_list[_i] += 1
	return in_dim_list


def list_sum(x):
	if len(x) == 1:
		return x[0]
	else:
		return x[0] + list_sum(x[1:])


def count_parameters(model):
	#total_params = sum([p.data.nelement() for p in model.parameters()])
	total_params = 0
	for module in model.modules():
		if type(module) in [nn.Conv2d, nn.Linear]:	
			total_params += module.weight.data.nelement()
			if type(module) in [nn.Linear]:
				total_params += module.bias.data.nelement()
		elif type(module) in [Q.QuantConv2d, Q.QuantLinear]:
			total_params += module.weight.data.nelement()/16 + 2
			if type(module) in [Q.QuantLinear]:
				total_params += module.bias.data.nelement()
		elif type(module) in [nn.BatchNorm2d]:
			total_params += module.weight.data.nelement()
			total_params += module.bias.data.nelement()
	return total_params


def count_conv_flop(layer, x):
	out_h = int(x.size()[2] / layer.stride[0])
	out_w = int(x.size()[3] / layer.stride[1])
	delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
	            layer.kernel_size[1] * out_h * out_w / layer.groups
	return delta_ops


def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""

	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class BasicUnit(nn.Module):

	def forward(self, x):
		raise NotImplementedError

	@property
	def unit_str(self):
		raise NotImplementedError

	@property
	def config(self):
		raise NotImplementedError

	@staticmethod
	def build_from_config(config):
		raise NotImplementedError

	def get_flops(self, x):
		raise NotImplementedError

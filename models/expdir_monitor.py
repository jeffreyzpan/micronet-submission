"""
Input: 'path' to the folder that contains configs and weights of a network
	Follow the configs and train the network
Output: 'Results' after training
"""

import json
import os

import torch

from models.networks.run_manager import RunConfig, RunManager
from models.networks import get_net_by_name


class ExpdirMonitor:

	def __init__(self, expdir, dataset='cifar100'):
		self.expdir = os.path.realpath(expdir)
		os.makedirs(self.expdir, exist_ok=True)
		self.dataset=dataset

	""" expdir paths """

	@property
	def logs_path(self):
		return '%s/logs' % self.expdir

	@property
	def save_path(self):
		return '%s/checkpoint' % self.expdir

	@property
	def output_path(self):
		return '%s/output' % self.expdir

	@property
	def run_config_path(self):
		return '%s/run.config' % self.expdir

	@property
	def net_config_path(self):
		return '%s/net.config' % self.expdir

	@property
	def init_path(self):
		return '%s/init' % self.expdir

	""" methods for loading """

	def load_run_config(self, print_info=False, dataset='cifar10'):
		if os.path.isfile(self.run_config_path):
			run_config = json.load(open(self.run_config_path, 'r'))
		else:
			run_config = RunConfig.get_default_run_config(dataset)
		run_config = RunConfig(**run_config)
		if print_info:
			print('Run config:')
			for k, v in run_config.config.items():
				print('\t%s: %s' % (k, v))
		return run_config

	def load_net(self, print_info=False, quantize=False):
		assert os.path.isfile(self.net_config_path), 'No net configs found in <%s>' % self.expdir
		net_config_json = json.load(open(self.net_config_path, 'r'))
		if print_info:
			print('Net config:')
			for k, v in net_config_json.items():
				if k != 'blocks':
					print('\t%s: %s' % (k, v))
		net = get_net_by_name(net_config_json['name']).build_from_config(net_config_json, quantize=quantize)
		return net

	def load_init(self):
		if os.path.isfile(self.init_path):
			if torch.cuda.is_available():
				checkpoint = torch.load(self.init_path)
			else:
				checkpoint = torch.load(self.init_path, map_location='cpu')
			return checkpoint
		else:
			return None

	""" methods for running """
	def run(self, quantize=False, quantize_dw=False, train=True, is_test=True, valid_size=None, resume=False):
		init = self.load_init()
		dataset=self.dataset
		#dataset = 'cifar10' if init is None else init.get('dataset', 'C10+')
		print(dataset)
		run_config = self.load_run_config(print_info=True, dataset=dataset)
		if valid_size is not None:
			run_config.valid_size = valid_size

		net = self.load_net(print_info=True, quantize=quantize)
		run_manager = RunManager(self.expdir, net, run_config, out_log=True)
		run_manager.save_config(print_info=True)

		if resume or quantize_dw:
			run_manager.load_model()
		elif init is not None:
			run_manager.net.module.load_state_dict(init['state_dict'], strict=False)
		if quantize_dw:
			run_manager.quantize()

		if train:
			run_manager.train()
			run_manager.save_model()

		loss, acc1 = run_manager.validate(is_test=is_test)
		if is_test:
			log = 'test_loss: %f\t test_acc1: %f' % (loss, acc1)
			run_manager.write_log(log, prefix='test')
			json.dump({'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1}, open(self.output_path, 'w'))
		else:
			log = 'valid_loss: %f\t valid_acc1: %f' % (loss, acc1)
			run_manager.write_log(log, prefix='valid')
			json.dump(
				{'valid_loss': '%f' % loss, 'valid_acc1': '%f' % acc1, 'valid_size': run_config.valid_size},
				open(self.output_path, 'w')
			)
		return acc1

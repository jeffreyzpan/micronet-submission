import os
import time
import shutil
import json
from datetime import timedelta
import math
import numpy as np

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from data_providers import get_data_provider_by_name
from models.utils import *
import models.modules.layers as layers


class RunConfig:

	def __init__(self, n_epochs=300, init_lr=0.1, lr_schedule_type='cosine', lr_schedule_param=None,
	             dataset='cifar10', train_batch_size=64, test_batch_size=100, valid_size=None, drop_last=True,
	             use_cutout=False, cutout_n_holes=1, cutout_size=16,
	             opt_type='sgd', opt_param=None, weight_decay=1e-4,
	             model_init='he_fout', init_div_groups=True,
	             validation_frequency=1, renew_logs=False, print_frequency=70):
		self.n_epochs = n_epochs
		self.init_lr = init_lr
		self.lr_schedule_type = lr_schedule_type
		self.lr_schedule_param = lr_schedule_param

		self.dataset = dataset
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.valid_size = valid_size
		self.drop_last = drop_last

		self.use_cutout = use_cutout
		self.cutout_n_holes = cutout_n_holes
		self.cutout_size = cutout_size

		self.opt_type = opt_type
		self.opt_param = opt_param
		self.weight_decay = weight_decay

		self.model_init = model_init
		self.init_div_groups = init_div_groups

		self.validation_frequency = validation_frequency
		self.renew_logs = renew_logs
		self.print_frequency = print_frequency

		self._data_provider = None

	@property
	def config(self):
		config = {}
		for key in self.__dict__:
			if not key.startswith('_'):
				config[key] = self.__dict__[key]
		return config

	def copy(self):
		return RunConfig(**self.config)

	@staticmethod
	def get_default_run_config(dataset='cifar10'):
		default_run_config = RunConfig()
		default_run_config.opt_param = {'momentum': 0.9, 'nesterov': True}
		default_run_config.dataset = dataset
		return default_run_config.config

	""" learning rate """

	def calc_learning_rate(self, epoch, batch=0, nBatch=None):
		param = {} if self.lr_schedule_param is None else self.lr_schedule_param
		if self.lr_schedule_type == 'cosine':
			T_total = self.n_epochs * nBatch
			T_cur = epoch * nBatch + batch
			lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
		else:
			reduce_stages = param.get('reduce_stages', [0.5, 0.75])
			reduce_factors = param.get('reduce_factors', [10, 10])
			lr = self.init_lr
			T_total = self.n_epochs * nBatch
			T_cur = epoch * nBatch + batch
			for _reduce_stage, _reduce_factor in zip(reduce_stages, reduce_factors):
				if T_cur >= _reduce_stage * T_total:
					lr /= _reduce_factor
		return lr

	def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
		""" adjust learning of a given optimizer and return the new learning rate """
		new_lr = self.calc_learning_rate(epoch, batch, nBatch)
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr
		return new_lr

	""" data providers """

	@property
	def data_provider(self):
		if self._data_provider is None:
			self._data_provider = get_data_provider_by_name(self.dataset, self.config)
		return self._data_provider

	@data_provider.setter
	def data_provider(self, val):
		self._data_provider = val

	@property
	def train_loader(self):
		return self.data_provider.train

	@property
	def valid_loader(self):
		return self.data_provider.valid

	@property
	def test_loader(self):
		return self.data_provider.test

	""" optimizer """

	def build_optimizer(self, net_params):
		if self.opt_type == 'sgd':
			opt_param = {} if self.opt_param is None else self.opt_param
			momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
			optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov,
			                            weight_decay=self.weight_decay)
		else:
			raise NotImplementedError
		return optimizer


class RunManager:

	def __init__(self, path, net, run_config: RunConfig, out_log=True):
		self.path = path
		self.net = net
		self.run_config = run_config
		self.out_log = out_log

		print(run_config.dataset)
		self._logs_path, self._save_path = None, None
		self.best_acc = 0
		self.start_epoch = 0

		# net info
		self.print_net_info()

		# initialize model (default)
		self.net.init_model(run_config.model_init, run_config.init_div_groups)

		# move network to GPU if available
		if torch.cuda.is_available():
			self.net = torch.nn.DataParallel(self.net).cuda()
			cudnn.benchmark = True

		# prepare criterion and optimizer
		if torch.cuda.is_available():
			self.criterion = nn.CrossEntropyLoss().cuda()
		else:
			self.criterion = nn.CrossEntropyLoss()
		self.optimizer = self.run_config.build_optimizer(self.net.parameters())

	""" network related paths """

	@property
	def save_path(self):
		if self._save_path is None:
			save_path = '%s/checkpoint' % self.path
			os.makedirs(save_path, exist_ok=True)
			self._save_path = save_path
		return self._save_path

	@property
	def logs_path(self):
		if self._logs_path is None:
			logs_path = '%s/logs' % self.path
			if self.run_config.renew_logs:
				shutil.rmtree(logs_path, ignore_errors=True)
			os.makedirs(logs_path, exist_ok=True)
			self._logs_path = logs_path
		return self._logs_path

	""" network information """

	def net_flops(self):
		data_shape = [1] + list(self.run_config.data_provider.data_shape)

		if isinstance(self.net, nn.DataParallel):
			net = self.net.module
			input_var = torch.autograd.Variable(torch.zeros(data_shape).cuda(), volatile=True)
		else:
			net = self.net
			input_var = torch.autograd.Variable(torch.zeros(data_shape), volatile=True)
		flop, _ = net.get_flops(input_var)
		return flop

	def print_net_info(self):
		# network architecture
		if self.out_log:
			print(self.net)

		# parameters
		total_params = count_parameters(self.net)
		if self.out_log:
			print('Total training params: %.2fM' % (total_params / 1e6))
		# flops
		flops = self.net_flops()
		if self.out_log:
			print('Total FLOPs: %.1fM' % (flops / 1e6))
		with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
			fout.write(json.dumps({
				'param': '%.2fM' % (total_params / 1e6),
				'flops': '%.1fM' % (flops / 1e6),
			}) + '\n')

	""" save and load network """
	def save_model(self, checkpoint=None, is_best=False, model_name=None):
		if checkpoint is None:
			checkpoint = {'state_dict': self.net.module.state_dict()}

		if model_name is None:
			model_name = 'checkpoint.pth.tar'

		checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
		latest_fname = os.path.join(self.save_path, 'latest.txt')
		model_path = os.path.join(self.save_path, model_name)
		with open(latest_fname, 'w') as fout:
			fout.write(model_path + '\n')
		torch.save(checkpoint, model_path)

		if is_best:
			best_path = os.path.join(self.save_path, 'model_best.pth.tar')
			shutil.copyfile(model_path, best_path)

	def load_model(self, model_fname=None):
		latest_fname = os.path.join(self.save_path, 'latest.txt')
		if model_fname is None and os.path.exists(latest_fname):
			with open(latest_fname, 'r') as fin:
				model_fname = fin.readline()
				if model_fname[-1] == '\n':
					model_fname = model_fname[:-1]
			model_fname = model_fname + '/checkpoint.pth.tar'
		try:
			if model_fname is None or not os.path.exists(model_fname):
				model_fname = '%s/checkpoint.pth.tar' % self.save_path
				with open(latest_fname, 'w') as fout:
					fout.write(model_fname + '\n')
			if self.out_log:
				print("=> loading checkpoint '{}'".format(model_fname))

			if torch.cuda.is_available():
				checkpoint = torch.load(model_fname)
			else:
				checkpoint = torch.load(model_fname, map_location='cpu')

			if 'epoch' in checkpoint:
				self.start_epoch = checkpoint['epoch'] + 1
			if 'best_acc' in checkpoint:
				self.best_acc = checkpoint['best_acc']
			if 'optimizer' in checkpoint:
				self.optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

			self.net.module.load_state_dict(checkpoint['state_dict'], strict=False)
			if self.out_log:
				print("=> loaded checkpoint '{}'".format(model_fname))

			# set new manual seed
			new_manual_seed = int(time.time())
			torch.manual_seed(new_manual_seed)
			torch.cuda.manual_seed_all(new_manual_seed)
			np.random.seed(new_manual_seed)
		except Exception as e:
			print(e)
			if self.out_log:
				print('fail to load checkpoint from %s' % self.save_path)

	def save_config(self, print_info=True):
		""" dump run_config and net_config to the model_folder """
		os.makedirs(self.path, exist_ok=True)
		net_save_path = os.path.join(self.path, 'net.config')
		json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4)
		if print_info:
			print('Network configs dump to %s' % self.save_path)

		run_save_path = os.path.join(self.path, 'run.config')
		json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
		if print_info:
			print('Run configs dump to %s' % run_save_path)

	""" training"""

	def write_log(self, log_str, prefix, should_print=True):
		""" prefix: valid, train, test """
		if prefix in ['valid', 'test']:
			with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
				fout.write(log_str + '\n')
				fout.flush()
		if prefix in ['valid', 'test', 'train']:
			with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
				if prefix in ['valid', 'test']:
					fout.write('=' * 10)
				fout.write(log_str + '\n')
				fout.flush()
		if should_print:
			print(log_str)

	def train(self):
		data_loader = self.run_config.train_loader

		for epoch in range(self.start_epoch, self.run_config.n_epochs):
			print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')

			batch_time = AverageMeter()
			data_time = AverageMeter()
			losses = AverageMeter()
			top1 = AverageMeter()

			# switch to train mode
			self.net.train()

			end = time.time()
			forward_time = 0
			for i, (_input, target) in enumerate(data_loader):
				data_time.update(time.time() - end)

				lr = self.run_config.adjust_learning_rate(self.optimizer, epoch, batch=i, nBatch=len(data_loader))

				if torch.cuda.is_available():
					target = target.cuda(non_blocking=True)
					_input = _input.cuda()

				# compute output
				output = self.net(_input)
				forward_time += time.time() - end
				loss = self.criterion(output, target)

				# measure accuracy and record loss
				acc1, _ = accuracy(output.data, target, topk=(1, 5))
				losses.update(loss.data.item(), _input.size(0))
				top1.update(acc1.item(), _input.size(0))

				# compute gradient and do SGD step
				self.net.zero_grad()  # or self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				# measure elapsed time
				batch_time.update(time.time() - end)
				end = time.time()

				if i % self.run_config.print_frequency == 0 or i + 1 == len(data_loader):
					batch_log = 'Train [{0}][{1}/{2}]\t' \
					            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
								'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
					            'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
					            'top 1-acc {top1.val:.3f} ({top1.avg:.3f})\tlr {lr:.5f}'. \
						format(epoch + 1, i, len(data_loader) - 1,
					           batch_time=batch_time, data_time=data_time, losses=losses, top1=top1, lr=lr)
					self.write_log(batch_log, 'train')
			time_per_epoch = batch_time.sum
			seconds_left = int((self.run_config.n_epochs - epoch - 1) * time_per_epoch)
			print('Time per epoch: %s, Est. complete in: %s' % (
				str(timedelta(seconds=time_per_epoch)),
				str(timedelta(seconds=seconds_left))))

			if (epoch + 1) % self.run_config.validation_frequency == 0:
				val_loss, val_acc = self.validate(is_test=False)
				is_best = val_acc > self.best_acc
				self.best_acc = max(self.best_acc, val_acc)
				val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop 1-acc {3:.3f} ({4:.3f})'. \
					format(epoch + 1, self.run_config.n_epochs, val_loss, val_acc, self.best_acc)
				self.write_log(val_log, 'valid')
			else:
				is_best = False

			self.save_model({
				'epoch': epoch,
				'best_acc': self.best_acc,
				'optimizer': self.optimizer.state_dict(),
				'state_dict': self.net.module.state_dict(),
			}, is_best=is_best)

	def validate(self, is_test=True, net=None, use_train_mode=False):
		if is_test:
			data_loader = self.run_config.test_loader
		else:
			data_loader = self.run_config.valid_loader

		if net is None:
			net = self.net

		if use_train_mode:
			net.train()
		else:
			net.eval()
		batch_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()

		end = time.time()
		with torch.no_grad():
			for i, (_input, target) in enumerate(data_loader):
				if torch.cuda.is_available():
					target = target.cuda(non_blocking=True)
					_input = _input.cuda()
				#input_var = torch.autograd.Variable(_input, volatile=True)
				#target_var = torch.autograd.Variable(target, volatile=True)

				# compute output
				output = net(_input)
				loss = self.criterion(output, target)

				# measure accuracy and record loss
				acc1, _ = accuracy(output.data, target, topk=(1, 5))
				losses.update(loss.data.item(), _input.size(0))
				top1.update(acc1.item(), _input.size(0))

				# measure elapsed time
				batch_time.update(time.time() - end)
				end = time.time()

				if i % self.run_config.print_frequency == 0 or i + 1 == len(data_loader):
					print('Test: [{0}/{1}]\t'
				      	'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				      	'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				      	'top 1-acc {top1.val:.3f} ({top1.avg:.3f})'.
				      	format(i, len(data_loader), batch_time=batch_time, loss=losses, top1=top1))
		return losses.avg, top1.avg
	def quantize(self):
		quantize_index = []
		for idx, module in enumerate(self.net.modules()):
			print(type(module))
			if type(module) in [nn.Conv2d, nn.Linear]:
				quantize_index.append(idx)
			print(idx, '->', module)
		print(quantize_index)
		

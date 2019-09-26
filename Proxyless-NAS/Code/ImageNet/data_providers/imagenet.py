import os

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import DataProvider


class ImagenetDataProvider(DataProvider):

	def __init__(self, save_path=None, train_batch_size=64, test_batch_size=100, valid_size=None, drop_last=False,
	             n_worker=20, distort_color=None, **kwargs):

		self._save_path = save_path

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		if distort_color == 'inception':
			# color distort in standard inception preprocessing
			color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
		else:
			color_transform = None

		if color_transform is None:
			train_transforms = transforms.Compose([
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			])
			print('no color jitter')
		else:
			train_transforms = transforms.Compose([
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				color_transform,
				transforms.ToTensor(),
				normalize,
			])
			print('use %s' % distort_color)

		train_dataset = datasets.ImageFolder(self.train_path, train_transforms)

		if valid_size is not None:
			if isinstance(valid_size, float):
				valid_size = int(valid_size * len(train_dataset))
			else:
				assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
			train_indexes, valid_indexes = self.random_sample_valid_set(
				[cls for _, cls in train_dataset.samples], valid_size, self.n_classes,
			)
			train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
			valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

			valid_dataset = datasets.ImageFolder(self.train_path, transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
			]))

			self.train = torch.utils.data.DataLoader(
				train_dataset, batch_size=train_batch_size, sampler=train_sampler,
				num_workers=n_worker, pin_memory=True, drop_last=drop_last,
			)
			self.valid = torch.utils.data.DataLoader(
				valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
				num_workers=n_worker, pin_memory=True, drop_last=False,
			)
		else:
			self.train = torch.utils.data.DataLoader(
				train_dataset, batch_size=train_batch_size, shuffle=True,
				num_workers=n_worker, pin_memory=True, drop_last=drop_last,
			)
			self.valid = None

		self.test = torch.utils.data.DataLoader(
			datasets.ImageFolder(self.valid_path, transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
			])), batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True, drop_last=False,
		)

		if self.valid is None:
			self.valid = self.test

	@staticmethod
	def name():
		return 'imagenet'

	@property
	def data_shape(self):
		return 3, 224, 224  # C, H, W

	@property
	def n_classes(self):
		return 1000

	@property
	def save_path(self):
		if self._save_path is None:
			self._save_path = '/ssd/dataset/imagenet'
		return self._save_path

	@property
	def data_url(self):
		raise ValueError('unable to download ImageNet')

	@property
	def train_path(self):
		return os.path.join(self.save_path, 'train')

	@property
	def valid_path(self):
		return os.path.join(self._save_path, 'val')


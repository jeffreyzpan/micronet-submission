import numpy as np


def get_split_list(in_dim, child_num):
	in_dim_list = [in_dim // child_num] * child_num
	for _i in range(in_dim % child_num):
		in_dim_list[_i] += 1
	return in_dim_list


class DataProvider:
	VALID_SEED = 0  # random seed for the validation set

	@staticmethod
	def name():
		""" Return name of the dataset """
		raise NotImplementedError

	@property
	def data_shape(self):
		""" Return shape as python list of one data entry """
		raise NotImplementedError

	@property
	def n_classes(self):
		""" Return `int` of num classes """
		raise NotImplementedError

	@property
	def save_path(self):
		""" local path to save the data """
		raise NotImplementedError

	@property
	def data_url(self):
		""" link to download the data """
		raise NotImplementedError

	@staticmethod
	def random_sample_valid_set(train_labels, valid_size, n_classes):
		train_size = len(train_labels)
		assert train_size > valid_size

		np.random.seed(DataProvider.VALID_SEED)  # set random seed before sampling validation set
		rand_indexes = np.random.permutation(train_size)

		train_indexes, valid_indexes = [], []
		per_class_remain = get_split_list(valid_size, n_classes)

		for idx in rand_indexes:
			label = train_labels[idx]
			if isinstance(label, float):
				label = int(label)
			elif isinstance(label, np.ndarray):
				label = np.argmax(label)
			else:
				assert isinstance(label, int)
			if per_class_remain[label] > 0:
				valid_indexes.append(idx)
				per_class_remain[label] -= 1
			else:
				train_indexes.append(idx)
		return train_indexes, valid_indexes

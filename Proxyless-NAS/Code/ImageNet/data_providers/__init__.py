from data_providers.imagenet import ImagenetDataProvider


def get_data_provider_by_name(name, train_params: dict):
	""" Return required data provider class """
	if name == ImagenetDataProvider.name():
		return ImagenetDataProvider(**train_params)
	else:
		print('Sorry, data provider for `%s` dataset '
		      'was not implemented yet' % name)
		exit()

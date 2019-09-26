from models.networks.ProxylessNASNets import ProxylessNASNets


def get_net_by_name(name):
	if name == ProxylessNASNets.__name__:
		return ProxylessNASNets
	else:
		raise ValueError('unrecognized type of network: %s' % name)

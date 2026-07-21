import torch.nn as nn


def weight_init(m):
	"""Custom weight initialization for TD-MPC2."""
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)

	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)

	elif isinstance(m, nn.ParameterList):
		for i,p in enumerate(m):
			if p.dim() == 3: # Linear
				nn.init.trunc_normal_(p, std=0.02) # Weight
				nn.init.constant_(m[i+1], 0) # Bias

	elif isinstance(m, nn.Conv3d):
		nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
		try:
			nn.init.constant_(m.bias, 0.001)
		except:
			pass

	elif isinstance(m, nn.Conv2d):
		nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
		try:
			nn.init.constant_(m.bias, 0.001)
		except:
			pass
	
	elif isinstance(m, nn.LSTM):
		for name, param in m.named_parameters():
			if 'weight' in name:
				nn.init.xavier_uniform_(param)
			elif 'bias' in name:
				nn.init.constant_(param, 0.0)

    # elif isinstance(m, nn.Linear):
    #     # nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #     nn.init.orthogonal_(m.weight, np.sqrt(2))
    #     nn.init.constant_(m.bias, 0.001)


def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)

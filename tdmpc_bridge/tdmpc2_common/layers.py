import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble

# use_base_mlp = False
class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		modules = nn.ModuleList(modules)
		fn, params, _ = combine_state_for_ensemble(modules)
		self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness='different', **kwargs)
		self.params = nn.ParameterList([nn.Parameter(p) for p in params])
		self._repr = str(modules)

	def forward(self, *args, **kwargs):
		return self.vmap([p for p in self.params], (), *args, **kwargs)

	def __repr__(self):
		return 'Vectorized ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div_(255.).sub_(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim
		# self.dim = sim_dim
	
	def forward(self, x):
		shp = x.shape
		# print(shp) 64 512
		x = x.view(*shp[:-1], -1, self.dim)
		# 64 64 8
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
	# def __init__(self, *args, dropout=0., act=nn.ReLU(inplace=True), **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
			
		return self.act(self.ln(x))
		# return self.act(x)

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"
	

class LinearWithLeakyReLU(nn.Linear):
    """
    Linear layer with LeakyReLU activation.
    """

    def __init__(self, *args, dropout=0., act=nn.LeakyReLU(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.act = act

    def forward(self, x):
        # 调用父类的 Linear 前向传播
        x = super().forward(x)
        # 应用 LeakyReLU 激活函数
        return self.act(x)

    def __repr__(self):
        return f"LinearWithLeakyReLU(in_features={self.in_features}, "\
               f"out_features={self.out_features}, "\
               f"bias={self.bias is not None}, "\
               f"act={self.act.__class__.__name__})"

class DynamicModelLSTM(nn.Module):
	def __init__(self, input_dim=514, hidden_dim=512, output_dim=512, num_layers=1):
		super(DynamicModelLSTM, self).__init__()

		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x, hidden=None):
		# print("lstm x:", x.shape)
		lstm_out, hidden = self.lstm(x, hidden)
		# print("lstm:", lstm_out.shape)
		output = self.fc(lstm_out[:, -1, :])
		# output = self.fc(lstm_out)
		return output, hidden
	
# 定义MLP结构
class Base_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(Base_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
			nn.LeakyReLU()
        )
        
    def forward(self, x):
        print("use base mlp \n")
        return self.model(x)



def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0., use_base_mlp=False, use_mish=False, use_lnrelu=False, use_s2l=False):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
		
	if use_base_mlp:
		print('==================>use base mlp now<======================')
		return Base_MLP(input_dim=in_dim, output_dim=out_dim).model
	
	else:
	# mlp_dims = [256, 256, 256]
		
		dims = [in_dim] + [184] + mlp_dims + [out_dim] if use_s2l else [in_dim] + mlp_dims + [out_dim]
		print("mlp dims", dims)
		mlp = nn.ModuleList()
		if use_lnrelu:
			print('==================>use ln+leakyrelu mlp now<======================')
			for i in range(len(dims) - 2):
				mlp.append(LinearWithLeakyReLU(dims[i], dims[i+1]))
			mlp.append(LinearWithLeakyReLU(dims[-2], dims[-1]) if act else nn.Linear(dims[-2], dims[-1]))	
		else:
			print('==================>use normln+mish mlp now<======================')
			for i in range(len(dims) - 2):
				mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
			if not use_mish:
				mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
			else:
				mlp.append(NormedLinear(dims[-2], dims[-1]) if act else nn.Linear(dims[-2], dims[-1]))
		return nn.Sequential(*mlp)
	


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			if cfg.use_act_enc:
				if cfg.use_multi_mlp_enc:
					if cfg.laser_goal_cat:
						if cfg.use_obs_multi_enc:
							out[k] = mlp(cfg.obs_dim + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], int(cfg.latent_dim/2), act=SimNorm(cfg), use_base_mlp=cfg.use_base_mlp, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu, use_s2l=cfg.use_s2l)		
						else:
							out[k] = mlp(cfg.obs_dim + cfg.goal_dim + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg), use_base_mlp=cfg.use_base_mlp, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu, use_s2l=cfg.use_s2l)	
					else:
						out[k] = mlp(cfg.obs_dim + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg), use_base_mlp=cfg.use_base_mlp, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu, use_s2l=cfg.use_s2l)
				else:	
					out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg), use_base_mlp=cfg.use_base_mlp, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu, use_s2l=cfg.use_s2l)
			else:
				if cfg.use_multi_mlp_enc:
					if cfg.laser_goal_cat:
						if cfg.use_obs_multi_enc:
							out[k] = mlp(cfg.obs_dim + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], int(cfg.latent_dim/2), use_base_mlp=cfg.use_base_mlp, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu, use_s2l=cfg.use_s2l)		
						else:
							out[k] = mlp(cfg.obs_dim + cfg.goal_dim + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, use_base_mlp=cfg.use_base_mlp, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu, use_s2l=cfg.use_s2l)	
					else:
						out[k] = mlp(cfg.obs_dim + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, use_base_mlp=cfg.use_base_mlp, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu, use_s2l=cfg.use_s2l)
				else:
					out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, use_base_mlp=cfg.use_base_mlp, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu, use_s2l=cfg.use_s2l)
		
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)

def enc_state(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			if cfg.use_act_enc:
				if cfg.laser_goal_cat:
					out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim - cfg.obs_dim - cfg.goal_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
				else:
					out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim - cfg.obs_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
			else:
				if cfg.laser_goal_cat:
					out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim - cfg.obs_dim - cfg.goal_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim)
				else:
					out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim - cfg.obs_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim)
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)

def enc_goal(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			if cfg.use_act_enc:
				out[k] = mlp(cfg.goal_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], int(cfg.latent_dim/2), act=SimNorm(cfg))
			else:
				out[k] = mlp(cfg.goal_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], int(cfg.latent_dim/2))
				
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def dynamic_model_LSTM(input_dim, hidden_dim, output_dim, num_layers):
	return DynamicModelLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)


# class DynamicModelLSTM(nn.Module):
#     def __init__(self, input_dim=514, hidden_dim=512, output_dim=512, num_layers=1):
#         super(DynamicModelLSTM, self).__init__()

#         # 定义 LSTM 网络
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

#         # 定义输出层
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, hidden=None):
		
# 		# 输入 x 的形状: (batch_size, seq_len, input_dim)

# 		# 通过 LSTM
# 		lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch_size, seq_len, hidden_dim)
		
# 		# 取最后一个时间步的输出进行预测
# 		output = self.fc(lstm_out[:, -1, :])  # output: (batch_size, output_dim)

# 		return output, hidden


# # 测试动态模型
# def test_dynamic_model():
#     batch_size = 8
#     seq_len = 1  # 单时间步输入
#     input_dim = 514

#     # 创建模型
#     model = DynamicModelLSTM(input_dim=input_dim)

#     # 创建假数据
#     x = torch.rand(batch_size, seq_len, input_dim)  # (batch_size, seq_len, input_dim)

#     # 前向传播
#     output, hidden = model(x)

#     print("Input shape:", x.shape)
#     print("Output shape:", output.shape)  # (batch_size, output_dim)

# if __name__ == "__main__":
#     test_dynamic_model()


# 利用当前激光信息隐变量连续预测下三帧激光信息状态
# def predict_next_frames():
#     batch_size = 8
#     seq_len = 8  # 历史八帧输入
#     input_dim = 514
#     num_future_frames = 3  # 连续预测三帧

#     # 创建模型
#     model = DynamicModelLSTM(input_dim=input_dim)

#     # 创建假数据
#     x = torch.rand(batch_size, seq_len, input_dim)  # (batch_size, seq_len, input_dim)

#     # 初始化 hidden 状态
#     hidden = (torch.zeros(1, batch_size, 512), torch.zeros(1, batch_size, 512))  # (num_layers, batch_size, hidden_dim)

#     # 前向传播获取第一个预测
#     outputs = []
#     output, hidden = model(x, hidden)
#     outputs.append(output)

#     # 使用第一个预测递归生成后续帧
#     for _ in range(num_future_frames - 1):
#         # 将上一帧的预测作为当前帧的输入
#         next_input = torch.cat([output, torch.zeros(batch_size, input_dim - 512)], dim=1).unsqueeze(1)  # 补全维度为514
#         output, hidden = model(next_input, hidden)
#         outputs.append(output)

#     # 将预测结果堆叠起来
#     predictions = torch.stack(outputs, dim=1)  # (batch_size, num_future_frames, 512)

#     print("Predictions shape:", predictions.shape)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.modules import fc, cnn, gnn, lstm, deepsets, set_transformer
from tools.utils import weights_init

####################################
#             Network utils
####################################
class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim_feature = 256 + 128 + 128
        self.dim_embedding = 128
        modes = ['fc', 'cnn', 'gnn_vanilla', 'lstm', 'deepsets', 'set_transformer']
        net_modules = [fc.Linear_Encoder, cnn.Image_Encoder, gnn.GNN_Encoder, lstm.LSTM_Encoder, deepsets.DeepSet_Encoder, set_transformer.SetTransformer]

        mode_idx = modes.index(args.encoder_type)
        self.deviation_mode = args.deviation_mode
        self.no_goal = args.no_goal
        self.progress_ratio_mode = args.progress_ratio_mode

        # 只要 surrounding 分支需要条件输入（hybrid / dual-branch / vit），
        # 就把 ego / goal 一起传给 Image_Encoder
        self.use_hybrid_attn_encoder = bool(
            args.encoder_type == 'cnn' and (
                getattr(args, 'use_hybrid_attn_encoder', False)
                or getattr(args, 'use_dual_branch_encoder', False)
                or getattr(args, 'image_encoder_backbone', 'cnn') == 'vit'
            )
        )

        self.surrounding_embedding = net_modules[mode_idx](args)

        if self.no_goal:
            self.ego_embedding = nn.Linear(2, self.dim_embedding)
        else:
            self.ego_embedding = nn.Linear(2, self.dim_embedding) if not self.deviation_mode else nn.Linear(2, int(self.dim_embedding/2))
            self.goal_embedding = nn.Linear(2, self.dim_embedding) if not self.deviation_mode else nn.Linear(2, int(self.dim_embedding/2))

        if self.deviation_mode:
            if self.progress_ratio_mode:
                self.progress_embedding = nn.Linear(3, int(self.dim_embedding/2))
                self.path_embedding = nn.Linear(2, int(self.dim_embedding/2))
            else:
                self.path_embedding = nn.Linear(2, self.dim_embedding)

        self.apply(weights_init)

    def forward(self, state):
        """
        state:
            state[0] -> ego vel [B, 2]
            state[1] -> goal / path [B, 2]
            state[2] -> surrounding
                        - 原始单分支/单分支hybrid: Tensor
                        - 双分支: dict {"combine": ..., "temporal": ...}
            state[3] -> progress/path (可选)
        """
        ego_state = F.leaky_relu(self.ego_embedding(state[0]))

        # surrounding 分支
        if self.use_hybrid_attn_encoder:
            cond_goal = None if self.no_goal else state[1]
            surrounding_state = self.surrounding_embedding(state[2], state[0], cond_goal)
        else:
            surrounding_state = self.surrounding_embedding(state[2])

        if self.no_goal:
            path_state = F.leaky_relu(self.path_embedding(state[1]))
            if self.progress_ratio_mode:
                progress_state = F.leaky_relu(self.progress_embedding(state[3]))
        else:
            goal_state = F.leaky_relu(self.goal_embedding(state[1]))
            if self.deviation_mode:
                path_state = F.leaky_relu(self.path_embedding(state[3]))

        if self.no_goal:
            if self.progress_ratio_mode:
                return torch.cat([ego_state, path_state, progress_state, surrounding_state], dim=1)
            else:
                return torch.cat([ego_state, path_state, surrounding_state], dim=1)
        else:
            # print(ego_state.shape, goal_state.shape, surrounding_state.shape)
            return torch.cat([ego_state, goal_state, surrounding_state], dim=1) \
                if not self.deviation_mode else torch.cat([ego_state, goal_state, path_state, surrounding_state], dim=1)


class FeatureMapper(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(FeatureMapper, self).__init__()
        self.fm = nn.Sequential(
            nn.Linear(dim_input, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, dim_output),
        )

    def forward(self, x):
        return self.fm(x)


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def save_models(net, optimizer, name, dir_path):
    torch.save(net.state_dict(), dir_path + "_" + name)
    torch.save(optimizer.state_dict(), dir_path + "_" + name + "_optimizer")
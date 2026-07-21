import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.distributions import MultivariateNormal
from ..policy import RL_Policy

from tools.buffer import ReplayBuffer_List
from tools.utils import weights_init, CostMap
from tools.utils_network import FeatureMapper, soft_update, save_models
from tdmpc2_common import layers, math, init

# TODO: Discrete


class Actor(nn.Module):
    def __init__(self, args, FeatureExtractor, logstd_min=-10, logstd_max=2):
        super(Actor, self).__init__()
        
        self.args = args
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max
        self.fe = FeatureExtractor(args) if not args.sac_mlp_enc else layers.enc(args)
        self.mean = FeatureMapper(self.fe.dim_feature, args.action_dim) if not args.sac_mlp_enc else FeatureMapper(args.latent_dim, args.action_dim)
        self.mean_af = nn.Tanh()
        self.std = FeatureMapper(self.fe.dim_feature, args.action_dim) if not args.sac_mlp_enc else FeatureMapper(args.latent_dim, args.action_dim)
        self.std_af = nn.Tanh()
        
        self.apply(weights_init).cuda()
    
    
    def forward(self, state):
        # state
        # print(state.shape)
        x = self.fe(state) if not self.args.sac_mlp_enc else self.fe[self.args.obs](state)
        # print("state shape: ", x.shape)
        # sac
        mean = self.mean_af(self.mean(x))
        logstd = self.std_af(self.std(x))
        logstd = 0.5 * ((self.logstd_max - self.logstd_min) * logstd + (self.logstd_max + self.logstd_min))

        cov = torch.diag_embed(torch.exp(logstd))
        dist = MultivariateNormal(mean, cov)
        u = dist.rsample()

        action = torch.tanh(u)
        logprob = dist.log_prob(u).unsqueeze(1) - torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        return action, logprob, mean


class Critic(nn.Module):
    def __init__(self, args, FeatureExtractor):
        super(Critic, self).__init__()
        self.args = args
        self.dim_action_feature = 64
        self.fe = FeatureExtractor(args) if not args.sac_mlp_enc else layers.enc(args)
        self.fm_1 = FeatureMapper(self.fe.dim_feature + self.dim_action_feature, 1) if not args.sac_mlp_enc else FeatureMapper(args.latent_dim + self.dim_action_feature, 1)
        self.fm_2 = FeatureMapper(self.fe.dim_feature + self.dim_action_feature, 1) if not args.sac_mlp_enc else FeatureMapper(args.latent_dim + self.dim_action_feature, 1)
        self.action_embedding = nn.Linear(2, 64)
        self.apply(weights_init).cuda()

    def forward(self, state, action):
        state = self.fe(state) if not self.args.sac_mlp_enc else self.fe[self.args.obs](state)
        state = torch.cat([state, F.leaky_relu(self.action_embedding(action))], 1)
        return self.fm_1(state), self.fm_2(state)

    def Q1(self, state, action):
        state = self.fe(state) if not self.args.sac_mlp_enc else self.fe[self.args.obs](state)
        state = torch.cat([state, F.leaky_relu(self.action_embedding(action))], 1)
        return self.fm_1(state)
    
    
class SAC(RL_Policy):
    def __init__(self, args, FeatureExtractor, discount=0.99, tau=0.005, lr_tune=1e-4, alpha_init = 1.0, target_entropy=None):
        super(SAC, self).__init__(args)

        self.actor = Actor(args, FeatureExtractor)
        self.critic = Critic(args, FeatureExtractor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer= torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        
        self.scheduler_actor = lr_scheduler.StepLR(self.actor_optimizer, 50, 0.99)
        self.scheduler_critic = lr_scheduler.StepLR(self.critic_optimizer, 50, 0.99)
        
        self.critic_loss = nn.MSELoss()

        # entropy tuning
        self.lr_tune = lr_tune
        self.alpha_init = alpha_init
        self.target_entropy = target_entropy
        if self.target_entropy == None:
            self.target_entropy = -np.prod((self.action_dim,)).item()
        
        self.log_alpha = torch.full((), np.log(self.alpha_init), requires_grad=True, dtype=torch.float32, device=torch.device('cuda'))
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_tune)

        self.discount = discount
        self.tau = tau
        self.args = args
        self.train_step = 0


    def update_buffer(self, state, action, next_state, reward, done, cbf_label, p_idx, a_idx):
        self.buffer.add(state, action, next_state, reward, done, cbf_label)
        

    def state_handler(self, state, if_batch=True):        
        if if_batch:   
            ego = torch.from_numpy(state[:, (self.surrouding_length):(self.surrouding_length+2)]).float().cuda() # B,2
            goal = torch.from_numpy(state[:, (self.surrouding_length+2):(self.surrouding_length+4)]).float().cuda() # B,2
            if self.info_type == 'state_based':
                surrouding = torch.from_numpy(state[:, :self.surrouding_length]).float().cuda()
            else:
                surrouding = torch.from_numpy(CostMap(state[:, :self.surrouding_length], 
                                                        self.laser_dim, self.laser_range, self.img_height, self.img_width, self.observation_dim, self.highlight_iterations,
                                                        combine=self.merge_vis, batch=True)).unsqueeze(1).float().cuda() # B,C,T,H,W
        else:
            ego = torch.from_numpy(state[(self.surrouding_length):(self.surrouding_length+2)]).unsqueeze(0).float().cuda() # B,2
            goal = torch.from_numpy(state[(self.surrouding_length+2):(self.surrouding_length+4)]).unsqueeze(0).float().cuda() # B,2
            if self.info_type == 'state_based':
                surrouding = torch.from_numpy(state[:self.surrouding_length]).unsqueeze(0).float().cuda()
            else:
                surrouding = torch.from_numpy(CostMap(state[:self.surrouding_length], 
                                                        self.laser_dim, self.laser_range, self.img_height, self.img_width, self.observation_dim, self.highlight_iterations,
                                                        combine=self.merge_vis)).unsqueeze(0).unsqueeze(1).float().cuda() # B,C,T,H,W
        
        return ego, goal, surrouding

            
    def action_handler(self, action):
        return torch.from_numpy(action).float().cuda() # B,2 
            
    @torch.no_grad()
    def select_action(self, state, step, if_train):
        state = self.state_handler(state, False) if not self.args.sac_mlp_enc else torch.from_numpy(state).unsqueeze(0).float().cuda()

        if if_train and step < self.random_exploration_length:
            action = torch.Tensor(1, self.action_dim).uniform_(-1, 1).cpu().numpy()
        else:
            action, _, _ = self.actor(state)
            action = action.detach().cpu().numpy()
    
        action_noise = (np.random.rand(self.action_dim) * 2. - 1.) * self.explore_noise if if_train else 0
        action = (action + action_noise).clip(-self.max_action, self.max_action)
        if self.train_step % 1e4 == 0:
            self.explore_noise = self.explore_noise * 0.99
            
        return action


    def train(self, writer, batch_size=16):
        self.train_step += 1
        if self.train_step < self.random_exploration_length:
            return

        self.actor.train()
        self.critic.train()

        '''load data batch'''
        state, action, next_state, reward, not_done, _ = self.buffer.sample(batch_size)

        # =====================================>>>  state preperation  <<<=====================================
        state = self.state_handler(state) if not self.args.sac_mlp_enc else torch.from_numpy(state).float().cuda()
        
        # =====================================>>>  next state preperation  <<<=====================================
        next_state = self.state_handler(next_state) if not self.args.sac_mlp_enc else torch.from_numpy(next_state).float().cuda()
        
        action = self.action_handler(action)
        reward = torch.from_numpy(reward).float().cuda() # B,1
        not_done = torch.from_numpy(not_done).float().cuda() # B,1
        
        '''critic'''
        with torch.no_grad():
            next_action, next_logprob, _ = self.actor(next_state)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_logprob
            target_q = reward + self.discount * not_done * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''actor'''
        action, logprob, _ = self.actor(state)
        actor_loss = (-self.critic.Q1(state, action) + self.alpha * logprob).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        '''automatic entropy tuning'''
        alpha_loss = self.log_alpha.exp() * (-logprob.mean() - self.target_entropy).detach()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        soft_update(self.critic_target, self.critic, self.tau)
        
        writer.add_scalar('Training/Policy/lr_actor', self.actor_optimizer.state_dict()['param_groups'][0]['lr'], self.train_step)
        writer.add_scalar('Training/Policy/lr_critic', self.critic_optimizer.state_dict()['param_groups'][0]['lr'], self.train_step)
        writer.add_scalar('Training/Policy/actor_loss', actor_loss.item(), self.train_step)
        writer.add_scalar('Training/Policy/critic_loss', critic_loss.item(), self.train_step)
        writer.add_scalar('Training/Policy/alpha', self.alpha.detach().item(), self.train_step)
        writer.add_scalar('Training/Policy/noise', self.explore_noise, self.train_step)
        
        if self.train_step % self.lr_scheduler_interval == 0:
            self.scheduler_actor.step()
            self.scheduler_critic.step()
        
        return self.train_step
                    


    def save(self, dir_path):
        save_models(self.critic, self.critic_optimizer, "critic", dir_path)
        save_models(self.actor, self.actor_optimizer, "actor", dir_path)


    def load(self, dir_path):
        self.critic.load_state_dict(torch.load(dir_path + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(dir_path + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(dir_path + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(dir_path + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)





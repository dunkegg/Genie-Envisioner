#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from ..policy import RL_Policy
from tools.buffer import ReplayBuffer_List
from tools.utils import weights_init, CostMap
from tools.utils_network import FeatureMapper, soft_update, save_models
from tdmpc2_common import layers, math, init
    
class Actor(nn.Module):
    def __init__(self, args, FeatureExtractor):
        super(Actor, self).__init__()

        self.fe = FeatureExtractor(args)
        self.fm = FeatureMapper(self.fe.dim_feature, args.action_dim)
        self.af = nn.Tanh()
        self.apply(weights_init).cuda()
    
    def forward(self, state):
        x = self.fe(state)
        return self.af(self.fm(x))


class Critic(nn.Module):
    def __init__(self, args, FeatureExtractor):
        super(Critic, self).__init__()

        self.dim_action_feature = 64
        self.fe = FeatureExtractor(args)
        self.fm_1 = FeatureMapper(self.fe.dim_feature + self.dim_action_feature, 1)
        self.fm_2 = FeatureMapper(self.fe.dim_feature + self.dim_action_feature, 1)
        self.action_embedding = nn.Linear(2, 64)
        self.apply(weights_init).cuda()

    def forward(self, state, action):
        state = self.fe(state)
        state = torch.cat([state, F.leaky_relu(self.action_embedding(action))], 1)
        return self.fm_1(state), self.fm_2(state)

    def Q1(self, state, action):
        state = self.fe(state)
        state = torch.cat([state, F.leaky_relu(self.action_embedding(action))], 1)
        return self.fm_1(state)


class TD3(RL_Policy):
    def __init__(self, args, FeatureExtractor, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.4, policy_freq=2):
        super(TD3, self).__init__(args)
        self.buffer = ReplayBuffer_List(args.buffer_size, args.state_dim, args.action_dim, args.action_type)
        self.actor = Actor(args, FeatureExtractor)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(args, FeatureExtractor)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        self.scheduler_actor = lr_scheduler.StepLR(self.actor_optimizer, 50, 0.99)
        self.scheduler_critic = lr_scheduler.StepLR(self.critic_optimizer, 50, 0.99)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.train_step = 0
        self.save_time = 0
                
        
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
        state = self.state_handler(state, False)
        if if_train and step < self.random_exploration_length:
            action = torch.Tensor(1, self.action_dim).uniform_(-1, 1).cpu().numpy()
        else:
            action = self.actor(state)
            action = action.detach().cpu().numpy()

        print('action before noise : \n', action)
        action_noise = (np.random.rand(self.action_dim) * 2. - 1.) * self.explore_noise if if_train else 0
        action = (action + action_noise).clip(-self.max_action, self.max_action)
        if self.train_step % 1e4 == 0:
            self.explore_noise = self.explore_noise * 0.999
            
        return action


    def train(self, writer, batch_size=16):
        self.train_step += 1
        if self.train_step < self.random_exploration_length:
            return
        self.actor.train()
        self.critic.train()

        '''
        load data
        '''
        state, action, next_state, reward, not_done, _ = self.buffer.sample(batch_size)

        # =====================================>>>  state preperation  <<<=====================================
        state = self.state_handler(state)
        
        # =====================================>>>  next state preperation  <<<=====================================
        next_state = self.state_handler(next_state)
        
        action = self.action_handler(action)
        reward = torch.from_numpy(reward).float().cuda() # B,1
        not_done = torch.from_numpy(not_done).float().cuda() # B,1
        
        # ===========================>> Image Test <<===========================
        # for m in range(batch_size):
        #     if not args.merge_vis:
        #         for i in range(self.sample_length):
        #             surrouding_all_out = Image.fromarray(255 * np.uint8(surrouding[m,0][i].detach().cpu().numpy()))
        #             surrouding_all_out.save('/home/cyx/img_test/test/' + str(self.save_time).zfill(7) + '.jpg')
        #             self.save_time += 1
        #     else:
        #         surrouding_all_out = Image.fromarray(255 * np.uint8(surrouding[m,0][0].detach().cpu().numpy()))
        #         surrouding_all_out.save('/home/cyx/img_test/test/' + str(self.save_time).zfill(7) + '.jpg')
        #         self.save_time += 1
        # ======================================================================

        '''
        critic
        '''
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''
        actor
        '''
        if self.train_step % self.policy_freq == 0:

            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)
            
            writer.add_scalar('Training/Policy/actor_loss', actor_loss.item(), self.train_step)
        
        writer.add_scalar('Training/Policy/lr_actor', self.actor_optimizer.state_dict()['param_groups'][0]['lr'], self.train_step)
        writer.add_scalar('Training/Policy/lr_critic', self.critic_optimizer.state_dict()['param_groups'][0]['lr'], self.train_step)
        writer.add_scalar('Training/Policy/critic_loss', critic_loss.item(), self.train_step)
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

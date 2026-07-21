#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from tools.utils import weights_init


class Seperate_Encoder(nn.Module):
    def __init__(self):
        super(Seperate_Encoder, self).__init__()

        names = self.__dict__
        self.agent_state_dim = int(params.S_DIM / params.NUM_AGENT)
        for num_idx in range(params.NUM_AGENT):
            names['linear_' + str(num_idx)] = nn.Sequential(
                nn.Linear(self.agent_state_dim, 64),
                nn.LeakyReLU(),
            ).cuda()
        self.linear_concat = nn.Linear(64 * params.NUM_AGENT, 256)
        
    def forward(self, state):
        state_list = []
        for num_idx in range(params.NUM_AGENT):
            temp_state = eval('self.linear_' + str(num_idx))(state[:, (self.agent_state_dim * num_idx):(self.agent_state_dim * (num_idx + 1))])
            state_list.append(temp_state)
        state = torch.cat(state_list, 1)
        state = self.linear_concat(state)        
        return state


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.group_linear = nn.Sequential(
            nn.Linear(params.S_DIM, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
    
    def forward(self, state):
        state = self.group_linear(state)        
        return state
    
    
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.embedding = Encoder()
        # self.embedding = Seperate_Encoder()
        
    def forward(self, state):
        state = self.embedding(state)
        return state


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
    
    
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        
        self.fe = FeatureExtractor()
        self.fm = FeatureMapper(256, 2 * params.NUM_AGENT)
        self.apply(weights_init).cuda()

    def forward(self, state):
        state = self.fe(state)
        return torch.tanh(self.fm(state)) 


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.fe = FeatureExtractor()
        self.action_embedding = nn.Linear(2 * params.NUM_AGENT, 64)
        self.fm_1 = FeatureMapper(256 + 64, 1)
        self.fm_2 = FeatureMapper(256 + 64, 1)
        self.apply(weights_init).cuda()
        
    def forward(self, state, action):
        state = self.fe(state)
        state = torch.cat([state, F.leaky_relu(self.action_embedding(action))], 1)
        return self.fm_1(state), self.fm_2(state)

    def Q1(self, state, action):
        state = self.fe(state)
        state = torch.cat([state, F.leaky_relu(self.action_embedding(action))], 1)
        return self.fm_1(state)
    
    
class TD3(object):
    def __init__(self, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.4, policy_freq=2):
        
        self.actor = Actor()
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params.LR_ACTOR)

        self.critic = Critic()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params.LR_CRITIC)

        self.scheduler_actor = lr_scheduler.StepLR(self.actor_optimizer, 50, 0.99)
        self.scheduler_critic = lr_scheduler.StepLR(self.critic_optimizer, 50, 0.99)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.critic_loss_viz = 0
        self.actor_loss_viz = 0

        self.total_it = 0
        self.save_time = 0
                
        
    def state_interpreter(self, state, if_batch=True):
        if if_batch: 
            return torch.from_numpy(state).float().cuda()
        else:
            return torch.from_numpy(state).unsqueeze(0).float().cuda()


    def select_action(self, state):
        state = self.state_interpreter(state, False)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state) # (B, A)
        return action.reshape(params.NUM_AGENT, params.A_DIM).detach()



    def train(self, replay_buffer, batch_size=16):
        self.actor.train()
        self.critic.train()
        self.total_it += 1

        state, action, next_state, reward, not_done, _ = replay_buffer.sample(batch_size)

        # =====================================>>>  state preperation  <<<=====================================
        state = self.state_interpreter(state)
        
        # =====================================>>>  next state preperation  <<<=====================================
        next_state = self.state_interpreter(next_state)
        
        action = torch.from_numpy(action).reshape(batch_size, -1).float().cuda()
        reward = torch.from_numpy(reward).float().cuda()
        not_done = torch.from_numpy(not_done).float().cuda()

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-params.MAX_ACTION, params.MAX_ACTION)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_loss_viz = critic_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:

            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_loss_viz = actor_loss.item()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return self.actor_loss_viz, self.critic_loss_viz

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


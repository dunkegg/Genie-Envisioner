import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from ..policy import RL_Policy

from tools.buffer import ReplayBuffer_List
from tools.utils import weights_init, CostMap
from tools.utils_network import FeatureMapper, soft_update, save_models


class Policy_Network(nn.Module):
    def __init__(self, args, FeatureExtractor):
        super(Policy_Network, self).__init__()

        self.fe = FeatureExtractor(args)
        self.fm = FeatureMapper(self.fe.dim_feature, len(args.discrete_actions))
        self.apply(weights_init).cuda()


    def forward(self, state):
        x = self.fe(state)
        return self.fm(x)
    
    
class DQN(RL_Policy):
    def __init__(self, args, FeatureExtractor, discount=0.99, tau=0.005, epsilon_prob = 0.9, decay_rate = 0.99999):
        super(DQN, self).__init__(args)
        self.buffer = ReplayBuffer_List(args.buffer_size, args.state_dim, args.action_dim, args.action_type)
        self.policy = Policy_Network(args, FeatureExtractor)
        self.policy_target = copy.deepcopy(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr_critic)
        self.discount = discount
        self.tau = tau
        self.epsilon_prob = epsilon_prob
        self.decay_rate = decay_rate
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 50, 0.99)
        
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
        return torch.from_numpy(action).long().cuda() # B,1
    
    
    def train(self, writer, batch_size=16):
        self.policy.train()
        self.train_step += 1
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

        self.epsilon_prob *= self.decay_rate

        with torch.no_grad():
            target_q = self.policy_target(next_state).max(dim=1, keepdim=True)[0]
            target_q = reward + self.discount * not_done * target_q

        current_q = self.policy(state)
        current_q = torch.gather(current_q, dim=1, index=action)
        
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        writer.add_scalar('Training/Policy/lr_actor', self.optimizer.state_dict()['param_groups'][0]['lr'], self.train_step)
        writer.add_scalar('Training/Policy/actor_loss', loss.item(), self.train_step)
        writer.add_scalar('Training/Policy/noise', self.explore_noise, self.train_step)
        
        if self.train_step % self.lr_scheduler_interval == 0:
            self.scheduler.step()
        
        return self.train_step
                    


    @torch.no_grad()
    def select_action(self, state):
        state = self.state_handler(state, False)
        if random.random() < self.epsilon_prob:
            action = torch.tensor(random.choice(range(len(self.discrete_actions)))).reshape(1,-1).cpu()
        else:
            action_value = self.policy(state)
            action = torch.argmax(action_value, dim=1).cpu()
        return action
    

    def save(self, dir_path):
        save_models(self.policy, self.optimizer, "policy", dir_path)


    def load(self, dir_path):
        self.policy.load_state_dict(torch.load(dir_path + "_policy"))
        self.optimizer.load_state_dict(torch.load(dir_path + "_policy_optimizer"))
        self.policy_target = copy.deepcopy(self.policy)




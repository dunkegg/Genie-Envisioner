
import copy
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.distributions import Categorical, MultivariateNormal
from ..policy import RL_Policy
from tools.buffer import RolloutBuffer_vanilla
from tools.utils import weights_init, CostMap
from tools.utils_network import FeatureMapper, soft_update, hard_update, save_models


class Policy_Discrete(nn.Module):

    def __init__(self, FeatureExtractor):
        super().__init__()

        self.fe = FeatureExtractor()
        self.actor = FeatureMapper(self.fe.dim_feature, params.DIS_A_DIM)
        self.af = nn.Softmax(dim=1)
        self.critic = FeatureMapper(self.fe.dim_feature, 1)
        self.apply(weights_init).cuda()

    def forward(self, state):
        x = self.fe(state)
        action_prob = self.af(self.actor(x))
        dist = Categorical(action_prob)
        action = dist.sample()
        return action.unsqueeze(1), action_prob, dist

    def evaluate(self, state, action):
        x = self.fe(state)
        action_prob = self.af(self.actor(x))
        value = self.critic(x)

        dist = Categorical(action_prob)
        logprob = dist.log_prob(action.squeeze()).unsqueeze(1)
        entropy = dist.entropy().unsqueeze(1)

        return logprob, value, entropy, dist


class Policy_Continuous(nn.Module):
    def __init__(self, args, FeatureExtractor, logstd_min=-2.5, logstd_max=1):
        super().__init__()

        self.fe = FeatureExtractor(args)
        self.mean = FeatureMapper(self.fe.dim_feature, args.action_dim)
        self.std = FeatureMapper(self.fe.dim_feature, args.action_dim)

        self.critic = FeatureMapper(self.fe.dim_feature, 1)
        self.apply(weights_init).cuda()
        
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max
        

    def forward(self, state):
        x = self.fe(state)
        mean = torch.tanh(self.mean(x))
        logstd = torch.tanh(self.std(x))
        logstd = (self.logstd_max - self.logstd_min) * logstd + (self.logstd_max + self.logstd_min)
        logstd *= 0.5

        cov = torch.diag_embed(torch.exp(logstd))
        dist = MultivariateNormal(mean, cov)
        action = dist.sample()
        return action, mean, dist
    

    def evaluate(self, state, action):
        x = self.fe(state)
        mean = torch.tanh(self.mean(x))
        logstd = torch.tanh(self.std(x))
        logstd = (self.logstd_max - self.logstd_min) * logstd + (self.logstd_max + self.logstd_min)
        logstd *= 0.5
        value = self.critic(x)

        cov = torch.diag_embed(torch.exp(logstd))
        dist = MultivariateNormal(mean, cov)
        logprob = dist.log_prob(action).unsqueeze(1)
        entropy = dist.entropy().unsqueeze(1)
        return logprob, value, entropy, dist


class PPO(RL_Policy):
    def __init__(self, args, FeatureExtractor, 
                 discount=0.99, epsilon_clip=0.2, sample_reuse=4,
                 weight_entropy = 0.01, weight_value = 0.5,
                 max_size=20000):
        super(PPO, self).__init__(args)
        
        self.discount = discount
        self.weight_value = weight_value
        self.weight_entropy = weight_entropy
        self.epsilon_clip = epsilon_clip
        self.sample_reuse = sample_reuse
        self.num_iters = int(max_size / args.batch_size) * self.sample_reuse

        if args.action_type == 'continuous':
            self.policy = Policy_Continuous(args, FeatureExtractor)
        else:
            self.policy = Policy_Discrete(FeatureExtractor)
        self.policy_old = copy.deepcopy(self.policy)
        self.optimizer = Adam(self.policy.parameters(), lr=args.lr_critic, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 50, 0.99)
        
        self.buffer = RolloutBuffer_vanilla(args.batch_size * 2, max_size, args.state_dim, args.action_dim, args.action_type, True, self.discount)
       
        
        self.step_train = 0
    
    
    def update_buffer(self, state, action, logprob, value, reward, done, cbf_label, p_idx, a_idx):
        
        self.buffer.add(state, action, logprob, value, reward, done, cbf_label)


    def train(self, writer, batch_size=16):    
        if len(self.buffer) < self.buffer._train_episode_length:
            return 
        
        print("================================")
        print("=======>  PPO TRAINING  <=======")
        print("================================")
        
        for _ in range(self.num_iters):
            self.step_train += 1
            self.buffer.reward2return_vanilla()
            state, action, reward = self.buffer.sample_vanilla()
            reward = (reward - reward.mean()) / (reward.std() + 1e-5)
            state = self.state_handler(state)
            action = self.action_handler(action)
            reward = torch.from_numpy(reward).float().cuda() # B,1
            
            
            with torch.no_grad():
                logprob_old, _, _, _ = self.policy_old.evaluate(state, action)
            logprob, value, entropy, _ = self.policy.evaluate(state, action)

            ratio = torch.exp(logprob - logprob_old)
            advantage = reward - value.detach()
            
            # PPO2
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantage

            loss_surr = -torch.min(surr1, surr2).mean()
            loss_entropy = -self.weight_entropy * entropy.mean()
            loss_value = self.weight_value * F.mse_loss(value, reward)
            loss = loss_surr + loss_entropy + loss_value
            
            writer.add_scalar('Training/Policy/loss', loss.detach().item(), self.step_train)
            writer.add_scalar('Training/Policy/loss_entropy', loss_entropy.detach().item(), self.step_train)
            writer.add_scalar('Training/Policy/loss_value', loss_value.detach().item(), self.step_train)
            writer.add_scalar('Training/Policy/loss_surr', loss_surr.detach().item(), self.step_train)
            writer.add_scalar('Training/Policy/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.step_train)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        hard_update(self.policy_old, self.policy)

        self.buffer.clear()
        return self.step_train

    
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
        # TODO:continuous / discrete
        return torch.from_numpy(action).float().cuda() # B,2
    

    @torch.no_grad()
    def select_action(self, state):
        state = self.state_handler(state, False)
        action, _, _ = self.policy_old(state)
        return action.detach().cpu().numpy(), [0], [0]
    
    
    def save(self, dir_path):
        save_models(self.policy_old, self.optimizer, "policy_old", dir_path)


    def load(self, dir_path):
        self.policy_old.load_state_dict(torch.load(dir_path + "_policy_old"))
        self.optimizer.load_state_dict(torch.load(dir_path + "_policy_old_optimizer"))
        self.policy_old_target = copy.deepcopy(self.policy_old)




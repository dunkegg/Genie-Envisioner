import copy
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.distributions import Normal, Categorical
import numpy as np

from ..policy import RL_Policy
from tools.buffer import RolloutBuffer
from tools.utils import weights_init, CostMap
from tools.utils_network import FeatureMapper, soft_update, hard_update, save_models


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# TODO: Policy_Discrete
class Policy_Discrete(nn.Module):
    def __init__(self, args, FeatureExtractor):
        super().__init__()

        self.fe = FeatureExtractor(args)
        self.critic = FeatureMapper(self.fe.dim_feature, 1)
        self.actor = FeatureMapper(self.fe.dim_feature, args.discrete_action_dim)
        self.apply(weights_init).cuda()

    def get_value(self, x):
        return self.critic(self.fe(x))

    def get_action_and_value(self, x, action=None):
        logits = self.actor(self.fe(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
            
        print(" ======> probs.log_prob(action) : (should be (B, 1))\n", probs.log_prob(action).shape)
        print(" ======> probs.entropy() : \n", probs.entropy().shape)
        return action, probs.log_prob(action), probs.entropy(), self.critic(self.fe(x))
        
        
class Policy_Continuous(nn.Module):
    def __init__(self, args, FeatureExtractor):
        super().__init__()

        self.fe = FeatureExtractor(args)
        self.critic = FeatureMapper(self.fe.dim_feature, 1)
        self.actor_mean = FeatureMapper(self.fe.dim_feature, args.action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, args.action_dim))
        self.apply(weights_init).cuda()
        
        # self.critic = nn.Linear(self.fe.dim_feature, 1)
        # layer_init(self.critic, std=1.0),
        # self.actor_mean = nn.Linear(self.fe.dim_feature, args.action_dim)
        # layer_init(self.actor_mean, std=0.01),
        # self.actor_logstd = nn.Parameter(torch.zeros(1, args.action_dim))
        # self.cuda()

    def get_value(self, x):
        return self.critic(self.fe(x))

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(self.fe(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # print(" ======> action_mean : \n", action_mean)
        # print(" ======> action_logstd : \n", action_logstd)
        # print(" ======> action_std : \n", action_std)
        
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1).unsqueeze(1), probs.entropy().sum(1).unsqueeze(1), self.critic(self.fe(x))
    
    
class PPO(RL_Policy):
    def __init__(self, args, FeatureExtractor):
        super(PPO, self).__init__(args)
        
        self.num_agent = args.num_agent
        self.num_update = args.num_update_ppo
        self.batch_size = args.batch_size
        self.clip_vloss = args.clip_vloss
        self.clip_coef = args.clip_coef
        self.norm_adv = args.norm_adv
        
        self.max_grad_norm = args.max_grad_norm
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.target_kl = args.target_kl
        
        
        if self.action_type == 'continuous':
            self.policy = Policy_Continuous(args, FeatureExtractor)
        else:
            print("=======================")
            self.policy = Policy_Discrete(args, FeatureExtractor)

        self.optimizer = Adam(self.policy.parameters(), lr=args.lr_critic, eps=1e-5)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 50, 0.99)
        self.on_policy_train_episode = args.on_policy_train_episode
        self.buffer = RolloutBuffer(args.batch_size, args.on_policy_train_episode, args.state_dim, args.action_dim, args.action_type, args.gae, args.gamma, args.gae_lamda)
        self.global_step = 0
        self.step_train = 0


    def update_buffer(self, state, action, logprob, value, reward, done, cbf_label, p_idx, a_idx):
        
        # None in logprob and value if using vanilla PPO
        data_idx = p_idx * self.num_agent + a_idx        
        self.buffer.add(state, action, logprob, value, reward, done, cbf_label, data_idx)
        self.global_step += 1
        
        
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
        if self.action_type == 'continuous':
            return torch.from_numpy(action).float().cuda() # B,2
        else:
            return torch.from_numpy(action).long().cuda() # B,1

    
    @torch.no_grad()
    def select_action(self, state):
        state = self.state_handler(state, False)
        action, logprob, _, value = self.policy.get_action_and_value(state)
        return action.detach().cpu().numpy(), logprob.detach().cpu().numpy(), value.flatten().detach().cpu().numpy()

        
    def train(self, writer, batch_size=16):    
        if len(self.buffer) < self.on_policy_train_episode:
            return
        
        print("================================")
        print("=======>  PPO TRAINING  <=======")
        print("================================")
        
        b_states, b_actions, b_logprobs, b_advantages, b_returns, b_values = self.buffer.sample_all_batch()
        b_size = b_states.shape[0]
        # =====================================>>>  data preperation  <<<=====================================
        b_states = self.state_handler(b_states)
        b_actions = self.action_handler(b_actions)
        b_logprobs = torch.from_numpy(b_logprobs).float().cuda() # B,1
        b_advantages = torch.from_numpy(b_advantages).float().cuda() # B,1
        b_returns = torch.from_numpy(b_returns).float().cuda() # B,1
        b_values = torch.from_numpy(b_values).float().cuda() # B,1 
        
        b_inds = np.arange(b_size)
        clipfracs = []
        
        for idx_update in range(self.num_update):
            print(" ======> idx_update : ", idx_update)
            np.random.shuffle(b_inds)
            for start in range(0, b_size, self.batch_size):
                if start == b_size-1:
                    break
                self.step_train += 1
                end = start + self.batch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value((b_states[0][mb_inds], b_states[1][mb_inds], b_states[2][mb_inds]), b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
    

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    # print(" ======> mb_advantages : \n", mb_advantages)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                
                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                writer.add_scalar('Training/Policy/loss', loss.detach().item(), self.step_train)
                writer.add_scalar('Training/Policy/loss_entropy', entropy_loss.detach().item(), self.step_train)
                writer.add_scalar('Training/Policy/loss_value', v_loss.detach().item(), self.step_train)
                writer.add_scalar('Training/Policy/loss_policy', pg_loss.detach().item(), self.step_train)
                writer.add_scalar('Training/Policy/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.step_train)
                writer.add_scalar("Training/Policy/old_approx_kl", old_approx_kl.item(), self.step_train)
                writer.add_scalar("Training/Policy/approx_kl", approx_kl.item(), self.step_train)
                writer.add_scalar("Training/Policy/clipfrac", np.mean(clipfracs), self.step_train)
    
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                writer.add_scalar("Training/Policy/explained_var", explained_var, self.step_train)
                
                if self.step_train % self.lr_scheduler_interval == 0:
                    self.scheduler.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        self.buffer.clear()
        return self.global_step
    
    
    def save(self, dir_path):
        save_models(self.policy, self.optimizer, "policy", dir_path)


    def load(self, dir_path):
        self.policy.load_state_dict(torch.load(dir_path + "_policy"))
        self.optimizer.load_state_dict(torch.load(dir_path + "_policy_optimizer"))

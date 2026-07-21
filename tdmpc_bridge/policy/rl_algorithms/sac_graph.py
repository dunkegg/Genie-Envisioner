import copy
from termcolor import cprint
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.distributions import MultivariateNormal
from ..policy import RL_Policy

from tools.utils import weights_init
from tools.utils_network import FeatureMapper, soft_update, save_models

from policy.modules.graph_pointer import GraphPointerPolicy, GraphQNet
from torch_geometric.data.batch import Batch


class SAC(RL_Policy):
    def __init__(self, args, discount=0.99, tau=0.005, lr_tune=5e-5, alpha_init = 1.0, target_entropy=None):
        super(SAC, self).__init__(args)

        
        self.actor = GraphPointerPolicy(node_dim=args.graph_node_feature_dim, 
                                        edge_dim=args.graph_edge_feature_dim,
                                        embedding_dim= args.graph_embedding_dim, 
                                        num_graph_padding=args.graph_num_graph_padding,
                                        n_layer_encoder=args.graph_n_layer_encoder, 
                                        n_layer_decoder=args.graph_n_layer_decoder,  
                                        n_head=args.graph_n_head, 
                                        n_layer_with_edge_attr=args.graph_n_layer_with_edge_attr,
                                        encoder_type=args.graph_encoder,
                                        using_relative_coords=args.graph_using_relative_coords).cuda()
        self.critic = GraphQNet(node_dim=args.graph_node_feature_dim, 
                                edge_dim=args.graph_edge_feature_dim,
                                embedding_dim= args.graph_embedding_dim, 
                                num_graph_padding=args.graph_num_graph_padding,
                                n_layer_encoder=args.graph_n_layer_encoder, 
                                n_layer_decoder=args.graph_n_layer_decoder,  
                                n_head=args.graph_n_head, 
                                n_layer_with_edge_attr=args.graph_n_layer_with_edge_attr,
                                encoder_type=args.graph_encoder,
                                using_relative_coords=args.graph_using_relative_coords).cuda()
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.graph_lr_actor)
        self.critic_optimizer= torch.optim.Adam(self.critic.parameters(), lr=args.graph_lr_critic)
        
        self.scheduler_actor = lr_scheduler.StepLR(self.actor_optimizer, 50, 0.99)
        self.scheduler_critic = lr_scheduler.StepLR(self.critic_optimizer, 50, 0.99)
        
        self.critic_loss = nn.MSELoss()
        
        self.greedy = args.graph_sac_greedy
        self.lr_scheduler_interval = args.lr_scheduler_interval

        # entropy tuning
        self.lr_tune = lr_tune
        self.alpha_init = alpha_init
        self.target_entropy = target_entropy
        if self.target_entropy == None:
            # self.target_entropy = -np.prod((args.graph_num_action_padding,)).item()
            self.target_entropy = 0.05 * (-np.log(1 / args.graph_num_action_padding))
            
        
        self.log_alpha = torch.full((), np.log(self.alpha_init), requires_grad=True, dtype=torch.float32, device=torch.device('cuda'))
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_tune)

        self.discount = discount
        self.tau = tau
        
        self.train_step = 0
        
        self.graph_num_action_padding = args.graph_num_action_padding

    def update_buffer(self, state, action, next_state, reward, done, p_idx):
        self.buffer.add(state, action, next_state, reward, done, p_idx)

    def state_handler(self, state, if_batch=True):    
        if self.graph_using_pyg:
            pyg_graph_abs = state['pyg_graph_abs'].cuda() if if_batch else Batch.from_data_list([state['pyg_graph_abs']]).cuda()
            pyg_graph_rel = state['pyg_graph_rel'].cuda() if if_batch else Batch.from_data_list([state['pyg_graph_rel']]).cuda()
            current_idx = state['current_idx'].cuda()
            action_idxes = state['action_idxes'].cuda()
            action_mask = state['action_mask'].cuda()
            
            return pyg_graph_abs, pyg_graph_rel, current_idx, action_idxes, action_mask
        else:
            node_info_padded = state['node_info_padded'].cuda()
            node_padding_mask = state['node_padding_mask'].cuda()
            edge_matrix = state['edge_matrix'].cuda()
            current_idx = state['current_idx'].cuda()
            action_idxes = state['action_idxes'].cuda()
            action_mask = state['action_mask'].cuda()
            
            return node_info_padded, node_padding_mask, edge_matrix, current_idx, action_idxes, action_mask

    def action_handler(self, action):
        return action.cuda() # B num_action_node_padding
            
    @torch.no_grad()
    def select_action(self, state, step, if_train):
        state = self.state_handler(state, False)
        
        if if_train and step < self.random_exploration_length:
            num_action = int(np.sum(state[-1].cpu().numpy()))
            action_index = np.random.choice(num_action, 1)
            all_action_indexes = state[-2].squeeze(-1).cpu().numpy()[0]
            action = all_action_indexes[action_index][0]
            action_attention = np.ones_like(action)
        else:
            action_attention = self.actor(state).detach()  
            # print("action mask : \n", state[-1])
            # print("action attention : \n", action)
            if self.greedy or (not if_train):
                action_index = torch.argmax(action_attention, dim=1).long()
            else:
                action_index = torch.multinomial(action_attention.exp(), 1).long().squeeze(1)

            action = state[-2][0, action_index.item()].cpu().numpy()
            action_index = action_index.cpu().numpy()
            action_attention = action_attention.cpu().numpy()
        return action, action_index, action_attention # idx in padding

    def train(self, writer, batch_size=16):
        self.train_step += 1
        if self.train_step < self.random_exploration_length:
            return
        elif self.train_step == self.random_exploration_length:
            cprint("========> Start Training !!!", color='green', attrs=['reverse', 'bold'])

        self.actor.train()
        self.critic.train()

        '''load data batch'''
        state, action, next_state, reward, not_done, _ = self.buffer.sample(batch_size)

        # =====================================>>>  state preperation  <<<=====================================
        # print("train state : nodes : \n", state[0].x)
        # print("train state : edges : \n", state[0].edge_index)
        # print("train state : batch : \n", state[0].batch)
        state = self.state_handler(state)
        
        # =====================================>>>  next state preperation  <<<=====================================
        next_state = self.state_handler(next_state)
        
        action = self.action_handler(action)
        reward = reward.unsqueeze(1).float().cuda() # B,1,1
        not_done = not_done.unsqueeze(1).float().cuda() # B,1,1
        
        '''critic'''
        with torch.no_grad():
            next_logprob = self.actor(next_state)
            # print("critic training => next_logprob : ", next_logprob.shape)
            target_q1, target_q2 = self.critic_target(next_state)
            
            # target_q = torch.min(target_q1, target_q2) - self.alpha * next_logprob
            next_q_values = torch.min(target_q1, target_q2)
            
            # print("critic training => next_q_values : ", next_q_values.shape)
            target_q = torch.sum(next_logprob.unsqueeze(2).exp() * (next_q_values - self.alpha * next_logprob.unsqueeze(2)), dim=1).unsqueeze(1)
            
            # print("critic training => next_logprob : ", next_logprob.shape)
            # print("critic training => target_q : ", target_q.shape)
            # print("critic training => not_done : ", not_done.shape)
            target_q = reward + self.discount * not_done * target_q
            # print("critic training => target_q : ", target_q.shape)
            
        all_q1, all_q2 = self.critic(state)
        # print("critic training => all_q1 : ", all_q1.shape)
        # print("critic training => all_q2 : ", all_q2.shape)
        # print("critic training => action : ", action.shape)
        # print("critic training => action : ", action)
        current_q1 = torch.gather(all_q1, 1, action.unsqueeze(-1))
        current_q2 = torch.gather(all_q2, 1, action.unsqueeze(-1))
        # print("critic training => current_q1 : ", current_q1.shape)
        # print("critic training => current_q2 : ", current_q2.shape)
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        # print("critic training => critic_loss : ", critic_loss.shape)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()   
  
            
        
        '''actor'''
        logprob = self.actor(state)
        # print("actor training => logprob : ", logprob.shape)
        # print("actor training => self.critic.Q1(state).detach() : ", self.critic.Q1(state).detach().shape)
        # actor_loss = (-self.critic.Q1(state) + self.alpha * logprob).mean()
        actor_loss = torch.sum((logprob.exp().unsqueeze(2) * (self.alpha * logprob.unsqueeze(2) - self.critic.Q1(state).detach())), dim=1).mean()
        # print("actor training => actor_loss : ", actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
            
        

        '''automatic entropy tuning'''
        # alpha_loss = self.log_alpha.exp() * (-logprob.mean() - self.target_entropy).detach()
        entropy = (logprob * logprob.exp()).sum(dim=-1)
        # print("actor training => logprob : ", logprob.shape)
        # print("actor training => logprob.exp() : ", logprob.exp().shape)
        # print("actor training => entropy : ", entropy.shape)
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
        # print("actor training => alpha_loss : ", alpha_loss)

        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        soft_update(self.critic_target, self.critic, self.tau)
        
        writer.add_scalar('Training/Policy/lr_actor', self.actor_optimizer.state_dict()['param_groups'][0]['lr'], self.train_step- self.random_exploration_length)
        writer.add_scalar('Training/Policy/lr_critic', self.critic_optimizer.state_dict()['param_groups'][0]['lr'], self.train_step- self.random_exploration_length)
        writer.add_scalar('Training/Policy/actor_loss', actor_loss.item(), self.train_step- self.random_exploration_length)
        writer.add_scalar('Training/Policy/critic_loss', critic_loss.item(), self.train_step- self.random_exploration_length)
        writer.add_scalar('Training/Policy/alpha_loss', alpha_loss.item(), self.train_step- self.random_exploration_length)
        writer.add_scalar('Training/Policy/alpha', self.alpha.detach().item(), self.train_step- self.random_exploration_length)
        writer.add_scalar('Training/Policy/entropy', entropy.detach().mean().item(), self.train_step- self.random_exploration_length)
        
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





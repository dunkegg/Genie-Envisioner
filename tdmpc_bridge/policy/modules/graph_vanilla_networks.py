import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops, softmax

from tools.utils_graph import GCNUnit, GATUnit, GCNEnhanceEncoder, GATEnhanceEncoder, PointerNet, EncoderMultiHeadAttention, DecoderMultiHeadAttention
from tools.utils_graph import padding_list, padding_graph, to_adjacency_matrix

import numpy as np


# ========================================> GCN Vanilla <========================================
class GCNPolicySoftmax(torch.nn.Module):
    """
    Function : 
        GCN policy network, softmax as output
    Input : 
        state : Batch, Graph(PyG), Input_Feature_Dim
        mask : Batch, 
    Output : 
        x : Num_Selected_Node, 
    """
    def __init__(self, input_dim, output_dim, embedding_dim, n_layer=1):
        super(GCNPolicySoftmax, self).__init__()
        self.input_dim, self.output_dim, self.embedding_dim = input_dim, output_dim, embedding_dim
        self.pre = GCNEnhanceEncoder(input_dim=input_dim, output_dim=embedding_dim, embedding_dim=embedding_dim, n_layer=n_layer)
        self.policy_layers = torch_geometric.nn.Sequential(
            "x, edge_index, batch_index", [
                # (nn.Dropout(p=0.5), "x -> xd"),
                (nn.Linear(embedding_dim, output_dim), "x -> x_out")
                ]
            ) 
        
    def forward(self, state, mask):
        x = self.pre(state)
        x = self.policy_layers(x, state.edge_index, state.batch)
        x = torch.masked_select(x.view(-1), mask)
        batch = torch.masked_select(state.batch, mask)
        x = softmax(x, batch) # Num_Selected_Node,
        return x


class GCNValueNet(torch.nn.Module):
    """
    Function : 
        GCN value function network
    Input : 
        state : Batch, Graph(PyG), Input_Feature_Dim
    Output : 
        x : Batch, 1
    """
    def __init__(self, input_dim, output_dim, embedding_dim, n_layer=1):
        super(GCNValueNet, self).__init__()
        self.input_dim, self.output_dim, self.embedding_dim = input_dim, output_dim, embedding_dim
        self.pre = GCNEnhanceEncoder(input_dim=input_dim, output_dim=embedding_dim, embedding_dim=embedding_dim, n_layer=n_layer)
        self.value_layers = torch_geometric.nn.Sequential(
            "x, edge_index, batch_index", [
                # (nn.Dropout(p=0.5), "x -> xd"),
                (global_mean_pool, "x, batch_index -> x1"),
                (nn.Linear(embedding_dim, output_dim), "x1 -> x_out")
                ]
            )
    
    def forward(self, state):
        x = self.pre(state)
        x = self.value_layers(x, state.edge_index, state.batch).mean(dim=1)
        return x


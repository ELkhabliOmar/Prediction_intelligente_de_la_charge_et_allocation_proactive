from policies.model.transformer_encoder.encoder import TransformerEncoder
from policies.model.transformer_encoder.multi_head_attention import MultiHeadAttention
import torch
import torch.nn as nn
import torch.functional as F

import random
import math



class TaskFormer(nn.Module):
    def __init__(self, d_in, d_pos, d_task, d_model=8, d_ff=8, n_heads=1, n_layers=1, dropout=0.1, mode="mixed"):
        super().__init__()

        
        self.nodes_embed = nn.Linear(d_in, d_model)
        self.task_embed = nn.Linear(d_task, d_model, bias=False)
        self.pos_nodes_embed = nn.Parameter(torch.zeros(d_pos, d_model))
        self.node_encoder = TransformerEncoder(d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.task_encoder = TransformerEncoder(d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        
        self.fc = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=1)


        self.mode = mode
        
        
    def forward(self, nodes, task, use_task=True):

        task = self.task_embed(task)
        nodes = self.nodes_embed(nodes)
        x = nodes + self.pos_nodes_embed 
        
        if (use_task and not self.mode == "node") or self.mode == "task":
            x = x + task.unsqueeze(1).repeat(1, nodes.size(1), 1)
        
        
        x = self.trasformer_encoder(x, None)

        x = self.fc(x)
        return x
        
    




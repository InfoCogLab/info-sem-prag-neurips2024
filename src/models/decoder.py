import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

import src.settings as settings

class Decoder(nn.Module):    
    def __init__(self, comm_dim, recons_dim, num_layers=3, num_imgs=1):
        super(Decoder, self).__init__()
        self.comm_dim = comm_dim
        self.recons_dim = recons_dim
        self.hidden_dim = 128
        self.num_imgs = num_imgs

        in_dim = comm_dim
        out_dim = self.hidden_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),  
                nn.ReLU(),               
                nn.Dropout(0.5)         
            ))
            in_dim = out_dim
            out_dim = self.hidden_dim
        # Then split into the number you need
        self.layers.append(nn.Linear(out_dim, self.hidden_dim * self.num_imgs))
        # And reconstruct each part separately.
        self.recons = nn.Linear(self.hidden_dim, recons_dim)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
        split_features = torch.reshape(x, (-1, self.num_imgs, self.hidden_dim))
        recons = self.recons(split_features)

        return recons

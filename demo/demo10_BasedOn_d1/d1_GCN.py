import pickle
from tqdm import tqdm
from torch_geometric.data import DataLoader
import random
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.metrics import roc_curve
import torch

class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCNClassifier, self).__init__()
        self.convs = nn.ModuleList()

        # self.bns = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels)) # *******************************************

            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            # self.bns.append(BatchNorm(hidden_channels))

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index)
            # x = self.bns[i](x)
            if i > 0:  # Skip connection for subsequent layers
                x += residual
            x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)
        return x.squeeze()



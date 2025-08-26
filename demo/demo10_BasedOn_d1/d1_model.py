import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
import warnings

warnings.filterwarnings("ignore")
import pickle
from tqdm import tqdm
from torch_geometric.data import DataLoader
import random
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from d1_GPSoc import *
from d1_GCN import *
from data import *


class ProcessGlobalFeatures(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(ProcessGlobalFeatures, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin1 = nn.Linear(in_dim, in_dim)
        self.lin2 = nn.Linear(in_dim, in_dim * 2)
        self.lin3 = nn.Linear(in_dim , out_dim)

        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim * 2)

        self.dropout = nn.Dropout(0.2)

    def forward(self, extra_feat):
#        extra_feat = self.lin1(extra_feat)
        extra_feat = self.bn1(extra_feat)
#        extra_feat = F.leaky_relu(extra_feat)
#        extra_feat = self.dropout(extra_feat)

#        extra_feat = self.lin2(extra_feat)
#        extra_feat = self.bn2(extra_feat)
#        extra_feat = F.leaky_relu(extra_feat)
#        extra_feat = self.dropout(extra_feat)

        extra_feat = self.lin3(extra_feat)
        return extra_feat




class FusionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 为每个元路径定义可学习权重γ
        self.gamma = nn.Parameter(torch.Tensor([0]))

    def forward(self, gcn_out, gat_out):
        # 论文公式(9)实现：加权融合
        normalized_gammas = torch.sigmoid(self.gamma)
        combined = normalized_gammas * gcn_out + (1 - normalized_gammas) * gat_out
        # print("权重：", normalized_gammas)
        return combined


class GATGCN(nn.Module):
    def __init__(self, node_dim, edge_input_dim, hidden_dim, global_dim, gcn_num_layers, gat_num_layers, dropout, device):
        super(GATGCN, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.gcn_num_layers = gcn_num_layers
        self.gat_num_layers = gat_num_layers
        self.dropout = dropout
        self.node_dim = node_dim
        self.edge_input_dim = edge_input_dim

        out_dim = 1

        self.add_module("FC_1", nn.Linear(hidden_dim + hidden_dim // 8, hidden_dim // 2, bias=True))
        self.add_module("dropout1", nn.Dropout(dropout))  # 第一个 Dropout 层
        self.add_module("FC_a", nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True))
        self.add_module("dropout2", nn.Dropout(dropout))  # 第二个 Dropout
        self.add_module("FC_2", nn.Linear(hidden_dim // 4, out_dim, bias=True))

        self.gcn_layers = GCNClassifier(node_dim, hidden_dim, out_dim, gcn_num_layers, dropout)
        self.gat_layers = GPSoc(node_dim, edge_input_dim, hidden_dim, gat_num_layers, dropout, device)
        self.fusion = FusionLayer()
        self.process_globalFeatures = ProcessGlobalFeatures(global_dim, hidden_dim // 8, dropout)

    def forward(self, data):

        X , node_feat, edge_index, batch = data.X, data.node_feat, data.edge_index, data.batch
        node_geo, h_E = get_geo_feat(X, edge_index)
        node_feat = torch.cat([node_feat, node_geo], dim=-1)
        gcn_out = self.gcn_layers(node_feat, edge_index, batch)
        gcn_out = torch.relu(gcn_out)
        gat_out = self.gat_layers(node_feat, h_E, edge_index, batch)
        gat_out = torch.relu(gat_out)
        h_V = self.fusion(gcn_out, gat_out)
        # print("h_V:", h_V.shape)

        extra_feat = data.extra_feat
        extra_feat = self.process_globalFeatures(extra_feat)
        h_V = torch.cat((h_V, extra_feat), dim=-1)
        # print("after cat:", h_V.shape)
        h_V = F.relu(self._modules["FC_1"](h_V))
        h_V = self._modules["dropout1"](h_V)

        h_V = F.relu(self._modules["FC_a"](h_V))
        h_V = self._modules["dropout2"](h_V)


        output = self._modules["FC_2"](h_V).sigmoid()
        return output.view([-1])



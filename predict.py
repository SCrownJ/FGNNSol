import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
import warnings
import pickle
from tqdm import tqdm
from torch_geometric.data import DataLoader
import random
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from model import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


set_seed(2024)

print("data loading...............")


prediction_path = "./dataset/prediction/pkl"

def load_data_with_names(path):
    data_list = []
    name_list = []
    # 确保路径存在
    if not os.path.exists(path):
        raise ValueError(f"路径不存在: {path}")

    for filename in os.listdir(path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f).to(torch.device('cuda'))
            data_list.append(data)
            protein_name = os.path.splitext(filename)[0]
            name_list.append(protein_name)

    if not data_list:
        warnings.warn(f"在路径 {path} 下未找到任何.pkl文件")

    return data_list, name_list


prediction_dataset, prediction_names = load_data_with_names(prediction_path)

batch_size = 16
prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)
print("data loaded !!!!!!!!!!")


def predictions(model, device, loader):
    model.eval()
    y_hat = torch.tensor([]).cuda()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            y_hat = torch.cat((y_hat, output), 0)
    return y_hat


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
node_dim = 1181 + 184
edge_in_channels = 450
extra_feat_dim = 16
hidden_channels = 256
dropout = 0
gcn_num_layers = 4
gat_num_layers = 2

model = FGNNSol(node_dim, edge_in_channels, hidden_channels, extra_feat_dim,
               gcn_num_layers, gat_num_layers, dropout, device).to(device)
model.load_state_dict(torch.load("./check_point/best_model/best_model.pt", map_location=device))
model.eval()
print("Model loaded successfully")


def get_predictions(model, device, loader):
    y_hat = predictions(model, device, loader)
    return y_hat.cpu().numpy().flatten()


predictions = get_predictions(model, device, prediction_loader)

result_df = pd.DataFrame({
    "protein": prediction_names,
    "prediction": predictions
})
result_df.to_csv("predictions.csv", index=False)
print(f"Predictions saved to predictions.csv，共{len(prediction_names)}条记录")

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
train_dataset = []
test_dataset = []
val_dataset = []
eval_dataset = []

train_path = "./dataset/train_data/pkl"
test_path = "./dataset/test_data/pkl"
val_path = "./dataset/val1_scere_data/pkl"
eval_path = "./dataset/eval_data/pkl"

for filename in os.listdir(train_path):
    file_path = os.path.join(train_path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f).to(torch.device('cuda'))
    train_dataset.append(data)

for filename in os.listdir(test_path):
    file_path = os.path.join(test_path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f).to(torch.device('cuda'))
    test_dataset.append(data)

for filename in os.listdir(val_path):
    file_path = os.path.join(val_path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f).to(torch.device('cuda'))
    val_dataset.append(data)

for filename in os.listdir(eval_path):
    file_path = os.path.join(eval_path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f).to(torch.device('cuda'))
    eval_dataset.append(data)

random.shuffle(train_dataset)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)  # 新增eval_loader
print("data loaded !!!!!!!!!!")


def train(model, device, loader, optimizer, criterion):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.float(), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def test(model, device, loader, criterion):
    model.eval()
    loss = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss += criterion(output, data.y).item()
    return loss / len(loader.dataset)


def predictions(model, device, loader):
    model.eval()
    y_hat = torch.tensor([]).cuda()
    y_true = torch.tensor([]).cuda()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            y_hat = torch.cat((y_hat, output), 0)
            y_true = torch.cat((y_true, data.y), 0)
    return y_hat, y_true


def binary_evaluate(y_true, y_hat, cut_off=0.5):
    binary_pred = [1 if pred >= cut_off else 0 for pred in y_hat]
    binary_true = [1 if true >= cut_off else 0 for true in y_true]
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    auc = metrics.roc_auc_score(binary_true, y_hat)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)
    print(
        f'Accuracy: {binary_acc:.8f}, Precision: {precision:.8f}, Recall: {recall:.8f}, '
        f'F1: {f1:.8f}, AUC: {auc:.8f}, MCC: {mcc:.8f}, '
        f'Sensitivity: {sensitivity:.8f}, Specificity: {specificity:.8f}'
    )


def evaluate_dataset(model, device, loader, dataset_name):

    loss = test(model, device, loader, criterion)

    y_hat, y_true = predictions(model, device, loader)

    r2 = metrics.r2_score(y_true.cpu(), y_hat.cpu())
    pearson = pearsonr(y_true.cpu(), y_hat.cpu())
    print(f'{dataset_name} loss: {loss:.8f}, R2: {r2:.8f}, Pearson: {pearson[0]:.8f}')

    y_hat_np = y_hat.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    print(f'{dataset_name} binary evaluation:')
    binary_evaluate(y_true_np, y_hat_np, cut_off=0.5)
    print('---')
    return y_hat_np, y_true_np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
node_dim = 1181 + 184
edge_in_channels = 450
extra_feat_dim = 16
hidden_channels = 256
dropout = 0
gcn_num_layers = 4
gat_num_layers = 2

model = FGNNSol(node_dim, edge_in_channels, hidden_channels, extra_feat_dim, gcn_num_layers, gat_num_layers, dropout,
                device).to(device)

for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)

initial_lr = 0.00001 
epochs = 80  # 训练轮数
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=initial_lr)

best_loss = float('inf')

print('Training start...............')
model.train()
lr = initial_lr
for epoch in range(1, epochs + 1):

    if epoch % 15 == 0:
        lr = lr * 0.75
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch > 65:
        lr = 0.1 * initial_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(model, device, train_loader, optimizer, criterion)

    train_loss = test(model, device, train_loader, criterion)
    eval_loss = test(model, device, eval_loader, criterion)
    test_loss = test(model, device, test_loader, criterion)
    val_loss = test(model, device, val_loader, criterion)

    if eval_loss < best_loss:
        best_loss = eval_loss
        torch.save(model.state_dict(), "./check_point/best_model.pt")

    print(
        f'Epoch: {epoch}, Train_Loss: {train_loss:.8f}, Eval_Loss: {eval_loss:.8f}, '
        f'Test_Loss: {test_loss:.8f}, ValLoss: {val_loss:.8f}'
    )

model.load_state_dict(torch.load("./check_point/best_model.pt"))
model.eval()

from sklearn import metrics
from scipy.stats import pearsonr

print("\nFinal evaluation results:")
eval_hat, eval_true = evaluate_dataset(model, device, eval_loader, "Eval")
test_hat, test_true = evaluate_dataset(model, device, test_loader, "Test")
val_hat, val_true = evaluate_dataset(model, device, val_loader, "Val")

binary_pred_test = [1 if pred >= 0.5 else 0 for pred in test_hat]
binary_true_test = [1 if true >= 0.5 else 0 for true in test_true]
fpr, tpr, thresholds = roc_curve(binary_true_test, test_hat)
df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
df.to_csv("./check_point/Roc.csv", index=False)
import os
import pandas as pd
import iFeatureOmegaCLI
import torch
import numpy as np
from networkx.algorithms.distance_measures import radius
from torch_geometric.data import Data
import warnings

warnings.filterwarnings("ignore")
from process_structure import *
import pickle
from tqdm import tqdm
import multiprocessing
import logging
from torch_geometric.utils import add_self_loops
import contextlib
import io
from Bio import pairwise2
import torch.nn.functional as F
import torch_geometric
from data import *


# 加载名称和标签的字典
def name_label_dict(path):
    pdb_chain_list = pd.read_csv(path, header=0)
    dict_pdb_chain = pdb_chain_list.set_index('gene')['solubility'].to_dict()
    return dict_pdb_chain


# 加载名称和序列的字典
def name_seq_dict(path):
    pdb_chain_list = pd.read_csv(path, header=0)
    dict_pdb_chain = pdb_chain_list.set_index('gene')['sequence'].to_dict()
    return dict_pdb_chain


# 处理单个蛋白质文件
def process_file(file):
    dict_path = '../dataset/csvFile/eSol_test.csv'
    name_dict = name_label_dict(dict_path)
    seq_dict = name_seq_dict(dict_path)

    # 设置文件路径
    pdb_directory = "../dataset/test_data/pdbTest"
    fasta_directory = "../dataset/test_data/fastaTest"
    pkl_directory = "../dataset/test_data/pkl"
    esmc_directory = "../dataset/test_data/esmc"  # 新增ESMC目录
    dssp_directory = "../dataset/test_data/dssp"  # 新增DSSP目录
    extra_feat_directory = "../dataset/test_data/extra_feat"

    try:
        # 基础路径验证
        pdb_path = os.path.join(pdb_directory, file + ".pdb")
        fasta_path = os.path.join(fasta_directory, file + "-model_v4.fasta")
        esmc_path = os.path.join(esmc_directory, file + ".npy")  # ESM特征路径
        dssp_path = os.path.join(dssp_directory, file + ".dssp")  # DSSP特征路径
        pkl_path = os.path.join(pkl_directory, file + "-model_v4.pkl")
        extra_feat_path = os.path.join(extra_feat_directory, file + ".npy")

        # 1. 从PDB文件提取原子坐标
        with open(pdb_path, "r") as f:
            pdb_lines = f.readlines()
        X = get_pdb_xyz(pdb_lines)
        X = torch.tensor(X, dtype=torch.float32)

        # 2. 加载预生成的ESM特征
        esm_emb = np.load(esmc_path).squeeze(0)
        esm_emb = esm_emb[1:-1]
        esm_emb = torch.tensor(esm_emb, dtype=torch.float32)

        # 3. 处理DSSP特征
        dssp_seq, dssp_matrix = process_dssp(dssp_path)
        # 序列对齐校验
        if dssp_seq != seq_dict[file]:
            dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq_dict[file])
        dssp_tensor = torch.tensor(np.array(dssp_matrix), dtype=torch.float32)

        # 4. 提取BLOSUM特征
        protein = iFeatureOmegaCLI.iProtein(fasta_path)
        # 重定向输出避免打印
        null_file = io.StringIO()
        with contextlib.redirect_stdout(null_file):
            protein.import_parameters('./Protein_parameters_train_setting.json')
        protein.get_descriptor("BLOSUM62")
        blosum_feat = torch.from_numpy((protein.encodings.values.reshape(-1, 20))).float()



        extra_feat = np.load(extra_feat_path)
        extra_feat = torch.tensor(extra_feat, dtype=torch.float32).unsqueeze(0)



        # 5. 生成半径图边索引
        radius = 10
        X_ca = X[:, 1]
        edge_index = torch_geometric.nn.radius_graph(X_ca, r=radius, loop=True, num_workers=4)

        # 6. 特征维度校验
        seq_len = len(seq_dict[file])
        assert blosum_feat.shape[0] == esm_emb.shape[0] == dssp_tensor.shape[0] == seq_len, \
            f"特征长度不一致: {file} BLOSUM({blosum_feat.shape[0]}) ESM({esm_emb.shape[0]}) DSSP({dssp_tensor.shape[0]}) SEQ({seq_len})"

        # 7. 组合节点特征
        node_features = torch.cat((blosum_feat, esm_emb, dssp_tensor), dim=1)

        # 8. 创建PyTorch Geometric数据对象
        label = torch.tensor(name_dict[file]).float().reshape(1, )
        data = Data(
            name=file,
            X=X,
            node_feat=node_features,
            edge_index=edge_index,
            extra_feat=extra_feat,
            y=label
        )

        # 添加自环
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=node_features.shape[0])

        # 保存为PKL文件
        with open(pkl_path, 'wb') as fpkl:
            pickle.dump(data, fpkl)

        print(f"成功处理 {file}")

    except Exception as e:
        logging.error(f"处理 {file} 时出错: {str(e)}")
        raise  # 调试时保留堆栈信息


def main():
    # 配置logging
    logging.basicConfig(
        filename='./protein_feature_extraction.log',
        level=logging.ERROR,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    dict_path = '../dataset/csvFile/eSol_test.csv'
    name_dict = name_label_dict(dict_path)
    file_names = list(name_dict.keys())

    print(f"开始为{len(file_names)}个蛋白质提取特征...")

    # 使用多进程并行处理
    with multiprocessing.Pool(processes=1) as pool:
        with tqdm(total=len(file_names)) as pbar:
            for _ in pool.imap_unordered(process_file, file_names):
                pbar.update()

    print("特征提取完成！")


if __name__ == '__main__':
    main()
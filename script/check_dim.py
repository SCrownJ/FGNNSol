import numpy as np

# 加载单个文件
file_path = "../dataset/train_data/extra_feat/aas.npy"  # 替换为实际文件名
data = np.load(file_path)
print(f"文件名: {file_path}")
print(f"形状: {data.shape} | 维度数: {data.ndim}")
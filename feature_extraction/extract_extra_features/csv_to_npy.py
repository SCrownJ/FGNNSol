import pandas as pd
import numpy as np
import os

# 读取CSV文件
df = pd.read_csv('../dataset/val2_371_data/PCF.csv')

# 设置输出目录并创建（如果不存在）
output_dir = '../dataset/val2_371_data/extra_feat'
os.makedirs(output_dir, exist_ok=True)

# 遍历每一行数据
for index, row in df.iterrows():
    # 提取文件名并去除.txt扩展名
    file_name = row['file_name']
    base_name = os.path.splitext(file_name)[0]  # 正确处理多后缀的情况

    # 提取特征数据（排除file_name列）
    features = row.drop('file_name').values.astype(np.float32)

    # 构建输出路径并保存
    save_path = os.path.join(output_dir, f"{base_name}.npy")
    np.save(save_path, features)

print("所有特征文件已成功保存。")
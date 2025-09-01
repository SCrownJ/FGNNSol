import pandas as pd
import os

# 定义输入 CSV 文件路径和输出文件夹路径
input_csv_path = '../dataset/csvFile/predict.csv'
output_folder = '../dataset/prediction/fasta'

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取 CSV 文件
data = pd.read_csv(input_csv_path)

# 遍历数据中的每一行
for index, row in data.iterrows():
    gene_name = row['name']
    sequence = row['sequence']

    # 定义输出文件路径
    output_file_path = os.path.join(output_folder, f"{gene_name}.fasta")

    # 写入 FASTA 文件
    with open(output_file_path, 'w') as file:
        file.write(f">{gene_name}\n")
        file.write(f"{sequence}\n")
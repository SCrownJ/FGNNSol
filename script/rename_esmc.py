import os

# 设置目录路径
dir_path = "../dataset/val2_371_data/esmc"

# 遍历目录下所有文件
for filename in os.listdir(dir_path):
    # 仅处理符合条件的.npy文件
    if filename.endswith(".npy") and "-model_v4" in filename:
        # 提取gene部分（去除"-model_v4"和扩展名）
        gene_part = filename.replace("-model_v4", "")
        new_filename = gene_part  # 新文件名已经是gene_part + .npy

        # 构建旧路径和新路径
        old_path = os.path.join(dir_path, filename)
        new_path = os.path.join(dir_path, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名成功: {filename} -> {new_filename}")

print("所有文件处理完成。")
import pandas as pd
import numpy as np
import os


def process_csv_to_npy(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        file_name = row['file_name']
        base_name = os.path.splitext(file_name)[0]
        features = row.drop('file_name').values.astype(np.float32)

        save_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(save_path, features)

    print(f"{csv_path} 已成功转换为 {output_dir} 中的npy文件")


if __name__ == "__main__":
    csv_output_pairs = [
        ("train_extra_feat.csv", "../../dataset/train_data/extra_feat"),
        ("test_extra_feat.csv", "../../dataset/test_data/extra_feat"),
        ("eval_extra_feat.csv", "../../dataset/eval_data/extra_feat"),
        ("val_extra_feat.csv", "../../dataset/val1_scere_data/extra_feat")
    ]

    for csv_path, output_dir in csv_output_pairs:
        if not os.path.exists(csv_path):
            print(f"警告：{csv_path} 文件不存在，已跳过处理")
            continue
        process_csv_to_npy(csv_path, output_dir)

    print("所有数据集的特征转换已完成")

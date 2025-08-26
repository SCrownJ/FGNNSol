import os
import csv
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData

# 定义Kyte-Doolittle亲水性标度
HYDROPHOBICITY_SCALE = ProtParamData.kd


# 手动实现脂肪族指数（Aliphatic Index）
def manual_aliphatic_index(sequence: str) -> float:
    """
    计算公式: AI = (X_Ala) + 2.9*(X_Val) + 3.9*(X_Ile + X_Leu)
    其中 X_aa 为氨基酸摩尔百分比
    """
    counts = {'A': 0, 'V': 0, 'I': 0, 'L': 0}
    for aa in sequence:
        if aa in counts:
            counts[aa] += 1
    total = len(sequence)
    if total == 0:
        return 0.0
    ala = (counts['A'] / total) * 100
    val = (counts['V'] / total) * 100 * 2.9
    ile_leu = (counts['I'] / total + counts['L'] / total) * 100 * 3.9
    return ala + val + ile_leu


def process_group(fasta_dir, csv_path):
    """处理单组fasta文件与CSV的对应关系"""
    # 读取现有CSV数据并初始化新字段
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames
        new_columns = [
            'molecular_weight', 'aromaticity', 'instability_index',
            'gravy', 'aliphatic_index', 'absolute_charge_per_residue',
            'hydrophilic_index'
        ]
        # 整合字段名，避免重复
        fieldnames = original_fieldnames + [col for col in new_columns if col not in original_fieldnames]
        rows = list(reader)
        # 为每行初始化新字段
        for row in rows:
            for col in new_columns:
                if col not in row:
                    row[col] = ''

    # 构建文件名映射（忽略.txt后缀，用于匹配FASTA文件）
    file_to_row = {row['file_name'].replace('.txt', ''): row for row in rows}

    # 处理每个.fasta文件
    for fasta_file in os.listdir(fasta_dir):
        if not fasta_file.endswith('.fasta'):
            continue

        # 提取基名（去除-model_v4和.fasta）
        base_name = os.path.splitext(fasta_file)[0].split('-model_v4')[0]
        if base_name not in file_to_row:
            print(f"警告：CSV中未找到 {base_name}.txt，已跳过")
            continue

        # 读取并清洗序列
        with open(os.path.join(fasta_dir, fasta_file), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sequence = ''.join(line.strip() for line in lines[1:]).upper()  # 忽略首行注释
            # 过滤非法字符（仅保留标准氨基酸）
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            sequence = ''.join([aa for aa in sequence if aa in valid_aa])
            if not sequence:
                print(f"错误：{fasta_file} 无有效序列")
                continue

        # 计算特征
        try:
            pa = ProteinAnalysis(sequence)

            # 1. 分子量
            mw = pa.molecular_weight()

            # 2. 芳香性
            aromaticity = pa.aromaticity()

            # 3. 不稳定性指数
            instability = pa.instability_index()

            # 4. 疏水性指数（GRAVY）
            gravy = pa.gravy()

            # 5. 脂肪族指数（手动计算）
            aliphatic = manual_aliphatic_index(sequence)

            # 6. 每残基电荷（pH 7.0）
            charge = pa.charge_at_pH(7.0) / len(sequence)

            # 7. 亲水性指数
            hydrophilic = sum(HYDROPHOBICITY_SCALE.get(aa, 0) for aa in sequence) / len(sequence)

            # 更新CSV行
            file_to_row[base_name].update({
                'molecular_weight': f"{mw:.2f}",
                'aromaticity': f"{aromaticity:.4f}",
                'instability_index': f"{instability:.2f}",
                'gravy': f"{gravy:.4f}",
                'aliphatic_index': f"{aliphatic:.2f}",
                'absolute_charge_per_residue': f"{charge:.4f}",
                'hydrophilic_index': f"{hydrophilic:.4f}"
            })
        except Exception as e:
            print(f"处理 {fasta_file} 失败: {str(e)}")
            continue

    # 写回更新后的CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已完成 {csv_path} 的特征更新")


if __name__ == "__main__":
    # 定义三组输入输出路径对应关系
    groups = [
        ("../../dataset/train_data/fastaTrain", "train_extra_feat.csv"),
        ("../../dataset/test_data/fastaTest", "test_extra_feat.csv"),
        ("../../dataset/val1_scere_data/fastaVal1_scere", "val_extra_feat.csv")
    ]

    # 依次处理每组数据
    for fasta_dir, csv_path in groups:
        # 检查路径是否存在
        if not os.path.exists(fasta_dir):
            print(f"警告：fasta目录 {fasta_dir} 不存在，跳过该组")
            continue
        if not os.path.exists(csv_path):
            print(f"警告：CSV文件 {csv_path} 不存在，跳过该组")
            continue
        # 处理当前组
        print(f"开始处理 {fasta_dir} -> {csv_path}")
        process_group(fasta_dir, csv_path)

    print("所有特征已成功添加到对应CSV！")
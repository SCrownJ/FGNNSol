import os
import re
import pandas as pd
from pathlib import Path


class LogFeatureExtractor:
    def __init__(self):
        # 定义特征提取规则
        self.patterns = {
            'atoms': r"(\d+) atoms",
            'residues': r"(\d+) residues",
            'surface_area': r"Surface area.*?= (\d+)",
            'volume': r"Enclosed volume.*?= ([\d.e+]+)",
            'sasa': r"Solvent accessible area.*?= ([\d.]+)",
            'hbonds': r"(\d+) hydrogen bonds found",
            'clashes': r"(\d+) clashes",
            'coulombic_mean': r"Coulombic values.*?mean ([-+]?\d+\.\d+)"
        }

        # 特征处理管道
        self.processors = {
            'volume': lambda x: float(x.replace('e+', 'e')),
            'coulombic_mean': float,
            'sasa': float,
            'atoms': int,
            'residues': int,
            'hbonds': int,
            'clashes': int,
            'surface_area': int
        }

    def parse_file(self, file_path):
        """解析单个日志文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        features = {'file_name': Path(file_path).name}
        for feat, pattern in self.patterns.items():
            match = re.search(pattern, content)
            if match:
                raw_value = match.group(1)
                # 应用数据清洗和类型转换
                features[feat] = self.processors[feat](raw_value)
            else:
                features[feat] = None

        # 计算衍生特征
        if features['surface_area'] and features['sasa']:
            features['hydrophilic_ratio'] = 1 - (features['surface_area'] / features['sasa'])

        return features


def batch_process(input_dir, output_csv):
    """批量处理日志目录"""
    extractor = LogFeatureExtractor()

    # 获取所有日志文件
    file_list = [os.path.join(input_dir, f)
                 for f in os.listdir(input_dir)
                 if f.endswith('.txt')]

    # 并行处理（使用线程池加速）
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(extractor.parse_file, file_list))

    # 生成数据框架
    df = pd.DataFrame(results)

    # 处理缺失值
    df.fillna({
        'hbonds': 0,
        'clashes': 0,
        'coulombic_mean': df['coulombic_mean'].mean()
    }, inplace=True)

    # 保存结果
    df.to_csv(output_csv, index=False)
    print(f"成功处理 {len(df)} 个文件，保存至 {output_csv}")


if __name__ == "__main__":
    # 定义输入输出路径对
    path_pairs = [
        ("./train_txt", "train_extra_feat.csv"),
        ("./test_txt", "test_extra_feat.csv"),
        ("./val_txt", "val_extra_feat.csv")
    ]

    # 批量处理每个目录
    for input_dir, output_csv in path_pairs:
        batch_process(input_dir, output_csv)

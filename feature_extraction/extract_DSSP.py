import os
import pandas as pd
import warnings
from tqdm import tqdm
import multiprocessing
import logging

# 导入处理结构的模块
from process_structure import process_dssp, match_dssp

warnings.filterwarnings("ignore")


# 获取名称和序列的字典
def name_seq_dict(path):
    pdb_chain_list = pd.read_csv(path, header=0)
    dict_pdb_chain = pdb_chain_list.set_index('name')['sequence'].to_dict()
    return dict_pdb_chain


# 处理单个蛋白质文件，仅生成DSSP文件
def process_file(file):
    dict_path = '../dataset/csvFile/predict.csv'
    seq_dict = name_seq_dict(dict_path)

    # 设置文件路径
    pdb_directory = "../dataset/prediction/pdb"
    dssp_directory = "../dataset/prediction/dssp"
    mkdssp_path = "./mkdssp.exe"

    # 确保输出目录存在
    os.makedirs(dssp_directory, exist_ok=True)

    try:
        pdb_path = os.path.join(pdb_directory, file + ".pdb")

        # 检查PDB文件是否存在
        if not os.path.exists(pdb_path):
            logging.error(f"PDB文件不存在：{pdb_path}")
            return

        # 生成DSSP文件
        dssp_output = os.path.join(dssp_directory, f"{file}.dssp")
        command = f'"{mkdssp_path}" -i "{pdb_path}" -o "{dssp_output}"'
        ret = os.system(command)
        if ret != 0:
            logging.error(f"生成DSSP失败，命令返回码：{ret}，文件：{file}")
            return  # 跳过后续处理

        # 验证并处理DSSP文件（保持原有DSSP处理逻辑）
        dssp_seq, dssp_matrix = process_dssp(dssp_output)
        if dssp_seq != seq_dict[file]:
            # 如果序列不匹配，仍保持原有匹配逻辑，但不保存结果
            match_dssp(dssp_seq, dssp_matrix, seq_dict[file])

        print(f"成功为{file}生成DSSP文件：{dssp_output}")

    except Exception as e:
        logging.error(f"处理{file}时出错: {str(e)}")


def main():
    # 配置logging
    logging.basicConfig(filename='./dssp_file_generation.log', level=logging.ERROR,
                        format='%(asctime)s %(levelname)s: %(message)s')

    dict_path = '../dataset/csvFile/predict.csv'
    seq_dict = name_seq_dict(dict_path)
    file_names = list(seq_dict.keys())

    print(f"开始为{len(file_names)}个蛋白质生成DSSP文件...")

    # 使用多进程并行处理
    with multiprocessing.Pool(processes=1) as pool:  # 可根据需要调整进程数

        with tqdm(total=len(file_names)) as pbar:  # 创建进度条
            for _ in pool.imap_unordered(process_file, file_names):
                pbar.update()  # 更新进度条
        pool.close()
        pool.join()

    print("DSSP文件生成完成！")


if __name__ == '__main__':
    main()

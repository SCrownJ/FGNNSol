import numpy as np
from Bio import pairwise2


########## Process PDB file ##########
# 功能：从PDB文件中提取每个氨基酸的原子坐标（N、CA、C、O）和侧链（R基团）的平均坐标。
# 输入：PDB文件行列表。   输出：一个形状为 (N, 5, 3) 的NumPy数组，其中 N 是氨基酸数，5代表5个关键原子（N、CA、C、O、R基团），3是三维坐标。
# 用于predict.py
import numpy as np

def get_pdb_xyz(pdb_lines):
    current_pos = -1000
    X = []
    current_aa = {}  # N, CA, C, O, R

    for line in pdb_lines:
        # 检测到新残基或TER时处理当前残基
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                # 主链原子完整性检查（来自第二版）
                if all(key in current_aa for key in ["N", "CA", "C"]):
                    # 处理缺失的O原子（来自第二版）
                    if "O" not in current_aa:
                        current_aa["O"] = current_aa["CA"]
                    # 计算R基团坐标（逻辑与第一版一致）
                    R_group = []
                    for atom in current_aa:
                        if atom not in ["N", "CA", "C", "O"]:
                            R_group.append(current_aa[atom])
                    if not R_group:
                        R_group = [current_aa["CA"]]
                    R_group = np.array(R_group).mean(0)
                    X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                current_aa = {}
            if line[0:4].strip() == "ATOM":
                current_pos = int(line[22:26].strip())

        # 收集原子坐标（来自第一版逻辑）
        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom != "H":
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz

    # 处理最后一个残基（来自第二版补充）
    if current_aa != {} and all(key in current_aa for key in ["N", "CA", "C"]):
        if "O" not in current_aa:
            current_aa["O"] = current_aa["CA"]
        R_group = []
        for atom in current_aa:
            if atom not in ["N", "CA", "C", "O"]:
                R_group.append(current_aa[atom])
        if not R_group:
            R_group = [current_aa["CA"]]
        R_group = np.array(R_group).mean(0)
        X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])

    return np.array(X)


########## Get DSSP ##########
# 功能：解析DSSP文件，提取残基的溶剂可及性（RSA）和二级结构（SS）特征。
# 输入：DSSP文件路径。
# 输出：元组 (seq, dssp_feature)，其中：seq 是氨基酸序列（如"ACDEF..."）。dssp_feature 是每个残基的特征列表，每个特征包含1个RSA值和8维的SS向量。
# 用于predict.py
def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(8)
        SS_vec[SS_type.find(SS)] = 1
        ASA = float(lines[i][34:38].strip())
        RSA = min(1, ASA / rASA_std[aa_type.find(aa)]) # relative solvent accessibility
        dssp_feature.append(np.concatenate((np.array([RSA]), SS_vec)))

    return seq, dssp_feature


# 功能：通过全局序列比对，将DSSP特征与参考序列对齐。
# 输入：seq：DSSP解析的原始序列。dssp：DSSP特征列表。ref_seq：参考序列（如目标蛋白的野生型序列）。
# 输出： 与参考序列（ref_seq）长度一致的列表，其中每个元素是经过对齐的 DSSP 特征向量（包含溶剂可及性和二级结构信息）
# 用于predict.py
def match_dssp(seq, dssp, ref_seq):
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    padded_item = np.zeros(9)

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


########## Match PDB and Sequence ##########
# 功能：检查PDB文件与蛋白质序列的一致性
# 输入：pdb_lines: PDB文件行列表，sequence: 蛋白质序列字符串
# 输出：一个布尔值，指示PDB文件与序列是否一致
def check_pdb_seq_consistency(pdb_lines, sequence):
    # 提取PDB文件中的残基
    residue_positions = set()
    residue_types = {}
    aa_3to1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 
        'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 
        'TRP': 'W', 'TYR': 'Y'
    }
    
    for line in pdb_lines:
        if line[0:4].strip() == "ATOM" and line[13:16].strip() == "CA":
            pos = int(line[22:26].strip())
            res_type = line[17:20].strip()
            residue_positions.add(pos)
            residue_types[pos] = aa_3to1.get(res_type, 'X')
    
    # 检查残基数量
    if len(residue_positions) != len(sequence):
        print(f"残基数量不匹配: PDB={len(residue_positions)}, 序列={len(sequence)}")
        return False
    
    # 按顺序比较残基类型
    sorted_positions = sorted(list(residue_positions))
    for i, pos in enumerate(sorted_positions):
        if i < len(sequence) and residue_types[pos] != sequence[i]:
            print(f"位置{i+1}的残基类型不匹配: PDB={residue_types[pos]}, 序列={sequence[i]}")
            # 容忍少量不匹配
            if i > 0 and i < len(sequence) - 1:
                continue
            return False
    
    return True

import os
from chimerax.core.commands import run, register, CmdDesc


# 定义主逻辑函数
def process_pqr(session):
    # 定义输入输出路径组合
    path_pairs = [
        ("./train_pqr", "./train_txt"),
        ("./test_pqr", "./test_txt"),
        ("./val_pqr", "./val_txt")
    ]

    # 处理每个路径组合
    for input_dir, output_dir in path_pairs:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有.pqr文件
        pqr_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".pqr"):
                    pqr_files.append(os.path.join(root, file))

        # 处理每个文件
        for pqr_path in pqr_files:
            base_name = os.path.basename(pqr_path)
            output_name = os.path.splitext(base_name)[0] + ".txt"
            output_path = os.path.join(output_dir, output_name)

            # 执行命令序列
            run(session, f"open {pqr_path}")
            run(session, "log clear")
            run(session, "select all")
            run(session, "surface")
            run(session, "mlp")
            run(session, "coulombic")
            run(session, "measure area #1")
            run(session, "measure sasa")
            run(session, "measure volume #1")
            run(session, "hbonds")
            run(session, "contacts")
            run(session, "clash")

            # 保存日志
            run(session, f"log save {output_path}")
            run(session, "close all")
            print(f"Processed: {os.path.basename(pqr_path)} -> {output_name}")


# 定义命令描述
cmd_desc = CmdDesc(
    synopsis="Process PQR files to extract features"
)


# 注册命令 (关键步骤)
def register_command(logger):
    register("pqr_processor", cmd_desc, process_pqr)


# 立即注册（必须调用）
register_command(session)

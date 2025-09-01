from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput
import pandas as pd
import numpy as np
import os

# 创建输出目录（如果不存在）
OUTPUT_DIR = "../dataset/prediction/esmc/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main(client: ESMCInferenceClient):
    # 读取CSV文件
    df = pd.read_csv("../dataset/csvFile/predict.csv")

    # 处理每个序列
    for _, row in df.iterrows():
        sequence = row["sequence"]
        name = row["name"]
        output_filename = os.path.join(OUTPUT_DIR, f"{name}.npy")

        # 初始化ESMC蛋白序列对象
        protein = ESMProtein(sequence=sequence)

        # 使用logits端点，使用bf16进行推理优化
        protein_tensor = client.encode(protein)  # 将序列转化为索引
        output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        assert isinstance(
            output, LogitsOutput
        ), f"LogitsOutput was expected but got {output}"
        assert output.logits is not None and output.logits.sequence is not None
        assert output.embeddings is not None and output.embeddings is not None
        print(
            f"Client returned logits with shape: {output.logits.sequence.shape} and embeddings with shape: {output.embeddings.shape}"
        )

        # 保存嵌入结果为npy文件
        np.save(output_filename, output.embeddings)


def raw_forward(model: ESMC):
    # 读取CSV文件
    df = pd.read_csv("../dataset/csvFile/predict.csv")

    # 处理每个序列
    for _, row in df.iterrows():
        sequence = row["sequence"]
        name = row["name"]
        output_filename = os.path.join(OUTPUT_DIR, f"{name}_raw.npy")

        protein = ESMProtein(sequence=sequence, padding=1700)
        sequences = [protein.sequence, protein.sequence]

        # 直接使用模型的示例
        input_ids = model._tokenize(sequences)
        output = model(input_ids)
        logits, embeddings, hiddens = (
            output.sequence_logits,
            output.embeddings,
            output.hidden_states,
        )
        print(
            f"Raw model returned logits with shape: {logits.shape}, embeddings with shape: {embeddings.shape} and hidden states with shape {hiddens.shape}"
        )

        # 保存原始模型的嵌入结果为npy文件
        np.save(output_filename, embeddings)


if __name__ == "__main__":
    model = ESMC.from_pretrained("esmc_600m")
    main(model)
    raw_forward(model)

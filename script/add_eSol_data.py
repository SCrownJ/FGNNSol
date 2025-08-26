import pandas as pd

# 文件路径配置
esol_path = r"D:\Programs\dev\workspace\Pworkspace\alphafold3\demoStation\GATSol\dataset\csvFile\eSol_train.csv"
es_path = r"D:\Programs\dev\workspace\Pworkspace\alphafold3\eS_train.csv"
output_dir = r"D:\Programs\dev\workspace\Pworkspace\alphafold3\\"

# 读取CSV文件
df_esol = pd.read_csv(esol_path)
df_es = pd.read_csv(es_path)

# 提取gene列的集合
esol_genes = set(df_esol['gene'])
es_genes = set(df_es['gene'])

# 找出eSol有但eS没有的gene
esol_unique = esol_genes - es_genes
# 找出eS有但eSol没有的gene
es_unique = es_genes - esol_genes

# 筛选对应行数据
esol_only_df = df_esol[df_esol['gene'].isin(esol_unique)]
es_only_df = df_es[df_es['gene'].isin(es_unique)]

# 输出文件
esol_only_df.to_csv(output_dir + "eSol_have_but_eS_not_train.csv", index=False)
es_only_df.to_csv(output_dir + "eS_have_but_eSol_not_train.csv", index=False)

print("操作完成！已生成两个差异文件。")
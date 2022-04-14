import pandas as pd

from src.config.config import HEAD, RELATION, TAIL


def extract_sample_from_knowledge_graph(in_file_path: str,
                                        out_file_path: str,
                                        head_column_name: str = HEAD,
                                        relation_column_name: str = RELATION,
                                        tail_column_name: str = TAIL,
                                        fraction: float = 0.05):
    # reading
    df = pd.read_csv(in_file_path,
                     sep="\t",
                     names=[head_column_name, relation_column_name, tail_column_name])
    print("\n >>> Before sampling:")
    print(df.shape)
    s1 = df[relation_column_name].value_counts(normalize=False)
    s2 = df[relation_column_name].value_counts(normalize=True)
    print(pd.concat([s1, s2], axis=1))
    # sampling
    df_sample = df.sample(frac=fraction).reset_index(drop=True)
    print("\n >>> After sampling:")
    print(df_sample.shape)
    s1_sample = df_sample[relation_column_name].value_counts(normalize=False)
    s2_sample = df_sample[relation_column_name].value_counts(normalize=True)
    print(pd.concat([s1_sample, s2_sample], axis=1))
    # writing
    df_sample.to_csv(out_file_path, sep="\t", index=False, header=False, encoding="utf-8")


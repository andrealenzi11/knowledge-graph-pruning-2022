import pandas as pd

from src.config.config import HEAD, RELATION, TAIL


def read_kg_tsv(in_path: str,
                head_column_name: str = HEAD,
                relation_column_name: str = RELATION,
                tail_column_name: str = TAIL) -> pd.DataFrame:
    """
        Read he tsv file that represents the Knowledge Graph and return the respective DataFrame object
    """
    df = pd.read_csv(in_path,
                     sep="\t",
                     names=[head_column_name, relation_column_name, tail_column_name],
                     encoding="utf-8").astype(str)
    df[head_column_name] = df[head_column_name].astype(str)
    df[relation_column_name] = df[relation_column_name].astype(str)
    df[tail_column_name] = df[tail_column_name].astype(str)
    df = df.reset_index(drop=True)
    print(df.info(memory_usage="deep"))
    print(df.shape)
    print(df.columns)
    return df

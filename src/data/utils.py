import io
import requests
import pandas as pd
import yaml


def parse_data_path(data_name):
    with open('./data/raw/data_path.yml', 'r') as f:
        data_path_dict = yaml.safe_load(f)[data_name]
    return data_path_dict


def read_csv_from_drive(url):
    r = requests.get(url)
    df = pd.read_csv(io.BytesIO(r.content), header=None)
    return df


def get_dist(df, col):
    """評価ネットワークからノード特徴量を作る

    Arguments:
        df {pd.DataFrame}
        col {str}

    Returns:
        pd.DataFrame
    """
    df_cnt = df.groupby(
        [col]+['rating'])['time'].count().unstack(1, fill_value=0)
    df_dist = pd.DataFrame(df_cnt.values / df_cnt.sum(1).values.reshape(-1, 1),
                           columns=df_cnt.columns,
                           index=df_cnt.index)
    return df_dist

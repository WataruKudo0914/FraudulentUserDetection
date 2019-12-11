"""
生の評価データ -> (Signed Network, 教師データ)
"""
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


if __name__ == '__main__':
    network_df = read_csv_from_drive(
        'https://drive.google.com/uc?id=1przExqwz7QSUCyqSsgez6glA6kaEMQix')
    pass

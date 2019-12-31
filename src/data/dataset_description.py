import pandas as pd
from pathlib import Path
from collections import defaultdict


def describe_dataset(data_name):
    data_dir = Path('./data/processed/') / data_name
    network_df = pd.read_csv(data_dir / 'network.csv')
    gt_df = pd.read_csv(data_dir / 'gt.csv')
    ret_dict = {}
    ret_dict['Nodes'] = len(set(network_df.iloc[:, 0]) |
                            set(network_df.iloc[:, 1]))
    ret_dict['Edges'] = len(network_df)
    ret_dict['Benign Users'] = (gt_df.label == 1).sum()
    ret_dict['Fraudulent Users'] = (gt_df.label == -1).sum()
    return ret_dict


def describe_all(data_names):
    description_dict = defaultdict(list)
    for data_name in data_names:
        data_dict = describe_dataset(data_name)
        description_dict['Dataset'].append(data_name)
        description_dict['Nodes'].append(data_dict['Nodes'])
        description_dict['Edges'].append(data_dict['Edges'])
        description_dict['Benign Users'].append(data_dict['Benign Users'])
        description_dict['Fraudulent Users'].append(
            data_dict['Fraudulent Users'])
    ret_df = pd.DataFrame(description_dict)
    return ret_df


if __name__ == '__main__':
    data_names = [
        'otc',
        'alpha',
        'amazon',
        'epinions',
        'amazon_app',
        'amazon_music',
        'amazon_home'
    ]
    result_df = describe_all(data_names)
    result_df.to_csv('./reports/データセット基本統計量.csv')

"""
ネットワークのうち，最初の何割かをとってくる
"""
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import numpy as np
from src.data.utils import get_dist
import itertools


def get_early_network(data_name, rate):
    # amazonの場合は処理が異なる
    raw_dir = Path('./data/raw') / data_name
    df_ = pd.read_csv(raw_dir / 'network.csv', index_col=0)
    df_.columns = ['id1', 'id2', 'rating', 'time', 'weight']
    df = df_.sort_values('time').iloc[:int(len(df_)*rate), :]
    if data_name in ['amazon', 'amazon_electronics', 'amazon_sports']:
        # amazonの場合だけ, rating==3のエッジを捨てる
        df = df.loc[df.weight != 0]
    else:
        pass
    # raw gt
    gt_df = pd.read_csv(raw_dir/'gt.csv', index_col=0)
    gt_df.columns = ['user_id', 'label']
    gt_df = gt_df.loc[gt_df.user_id.isin(set(df.id1) | set(df.id2))]
    # id変換
    label_encoder = LabelEncoder()
    label_encoder.fit(np.hstack((df.id1, df.id2, gt_df.user_id)))
    df['id1_'] = label_encoder.transform(df.id1)
    df['id2_'] = label_encoder.transform(df.id2)
    gt_df['node_id'] = label_encoder.transform(gt_df.user_id)
    all_node_features_id1 = get_dist(df_, 'id1')
    all_node_features_id2 = get_dist(df_, 'id2')
    if data_name in ['amazon', 'amazon_electronics', 'amazon_sports']:
        early_node_features_df = pd.concat(
            [get_dist(df_, 'id1', rating_cols=all_node_features_id1.columns),
             get_dist(df_, 'id2', rating_cols=all_node_features_id2.columns)],
            1).fillna(0).sort_index()
    else:
        early_node_features_df = pd.concat(
            [get_dist(df, 'id1_', rating_cols=all_node_features_id1.columns),
             get_dist(df, 'id2_', rating_cols=all_node_features_id2.columns)],
            1).fillna(0).sort_index()
    # 保存
    early_data_dir = Path('./data/processed/early') / f'{data_name}_{rate}'
    if not early_data_dir.exists():
        early_data_dir.mkdir()
    df[['id1_', 'id2_', 'weight']].to_csv(early_data_dir / 'network.csv')
    gt_df[['node_id', 'label']].to_csv(early_data_dir / 'gt.csv')
    np.save(arr=label_encoder.classes_,
            file=early_data_dir / 'label_encoder.npy')
    early_node_features_df.to_csv(early_data_dir / 'node_feature.csv',
                                  index=None)
    return df, gt_df, early_node_features_df, label_encoder


if __name__ == '__main__':
    rates = [0.1, 0.2, 0.3]
    data_names = ['alpha', 'otc', 'amazon', 'amazon_electronics']
    for data_name, rate in itertools.product(data_names, rates):
        get_early_network(data_name, rate)

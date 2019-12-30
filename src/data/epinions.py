import pandas as pd
import numpy as np
from src.data.utils import parse_data_path, get_dist
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def main():
    data_name = 'epinions'
    data_path_dict = parse_data_path(data_name)
    # 出力用フォルダ
    output_dir = Path('./data/processed/') / data_name
    if not output_dir.exists():
        output_dir.mkdir()
    # raw_network置き場
    raw_network_dir = Path('./data/raw/') / data_name
    if not raw_network_dir.exists():
        raw_network_dir.mkdir()
    # ラベル
    gt_df = pd.read_csv(data_path_dict['gt'])
    gt_df.columns = ['user_id', 'label']
    gt_df.to_csv(raw_network_dir / 'gt.csv')
    # network
    network_df = pd.read_csv(data_path_dict['network'])
    network_df.to_csv(raw_network_dir / 'network.csv')
    network_df, gt_df, label_encoder, node_features_df = convert_to_input(
        network_df, gt_df
    )
    # 出力
    network_df[['id1_', 'id2_', 'weight']].to_csv(
        output_dir / 'network.csv', index=None)
    gt_df[['node_id', 'label']].to_csv(
        output_dir / 'gt.csv', index=None)
    np.save(arr=label_encoder.classes_,
            file=output_dir / 'label_encoder.npy')
    node_features_df.to_csv(
        output_dir / 'node_feature.csv', index=None)
    pass


def convert_to_input(network_df, gt_df):
    gt_df = gt_df.loc[gt_df.user_id.isin(
        set(network_df.id1.unique()) | set(network_df.id2.unique()))]
    label_encoder = LabelEncoder()
    label_encoder.fit(np.hstack((network_df.id1,
                                 network_df.id2,
                                 gt_df.user_id)))
    network_df['id1_'] = label_encoder.transform(network_df.id1)
    network_df['id2_'] = label_encoder.transform(network_df.id2)
    gt_df['node_id'] = label_encoder.transform(gt_df.user_id)
    node_features_df = pd.concat([get_dist(network_df, 'id1_'), get_dist(
        network_df, 'id2_')], 1).fillna(0).sort_index()
    return network_df, gt_df, label_encoder, node_features_df


if __name__ == '__main__':
    main()

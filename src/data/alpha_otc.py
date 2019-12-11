import pandas as pd
import numpy as np
from src.data.preprocess import parse_data_path, get_dist
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def main(data_name):
    assert data_name in ['alpha', 'otc']
    data_path_dict = parse_data_path(data_name)
    # 出力用フォルダ
    output_dir = Path('./data/processed/') / data_name
    if not output_dir.exists():
        output_dir.mkdir()
    # ラベル
    gt_df = pd.read_csv(data_path_dict['gt'], header=None)
    gt_df.columns = ['user_id', 'label']
    # network
    network_df = pd.read_csv(data_path_dict['network'], header=None)
    network_df.columns = ['id1', 'id2', 'rating', 'time']
    network_df['weight'] = network_df.rating.map(lambda x: 1 if x > 0 else -1)
    # id変換
    label_encoder = LabelEncoder()
    label_encoder.fit(np.hstack((network_df.id1,
                                 network_df.id2,
                                 gt_df.user_id)))
    network_df['id1_'] = label_encoder.transform(network_df.id1)
    network_df['id2_'] = label_encoder.transform(network_df.id2)
    gt_df['node_id'] = label_encoder.transform(gt_df.user_id)
    node_features_df = pd.concat([get_dist(network_df, 'id1_'), get_dist(
        network_df, 'id2_')], 1).fillna(0).sort_index()
    # 出力
    network_df[['id1', 'id2', 'weight']].to_csv(
        output_dir / 'network.csv', index=None)
    gt_df[['node_id', 'label']].to_csv(
        output_dir / 'gt.csv', index=None)
    np.save(arr=label_encoder.classes_,
            file=output_dir / 'label_encoder.npy')
    node_features_df.to_csv(
        output_dir / 'node_feature.csv', index=None)


if __name__ == '__main__':
    main("otc")

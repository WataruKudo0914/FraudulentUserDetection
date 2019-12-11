import pandas as pd
import numpy as np
from src.data.utils import parse_data_path, get_dist
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def main():
    data_name = 'amazon'
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
    network_df.columns = ['user_id', 'product_id', 'rating', 'time']
    network_df['weight'] = network_df.rating.map(lambda x: (x-3)/2).round()
    # rating==3を除外する
    truncated_network = network_df.loc[network_df.weight != 0, [
        'user_id', 'product_id', 'weight']]
    # id変換
    label_encoder = LabelEncoder()
    label_encoder.fit(np.hstack((truncated_network.user_id,
                                 truncated_network.product_id,
                                 gt_df.user_id)))
    truncated_network['id1'] = label_encoder.transform(
        truncated_network.user_id)
    truncated_network['id2'] = label_encoder.transform(
        truncated_network.product_id)
    gt_df['node_id'] = label_encoder.transform(gt_df.user_id)
    # node features
    user_dist = get_dist(network_df, 'user_id')
    product_dist = get_dist(network_df, 'product_id')
    user_product_dist = pd.concat([user_dist, product_dist], 1).fillna(0)
    node_features_df = user_product_dist.loc[label_encoder.classes_]
    # 出力
    truncated_network[['id1', 'id2', 'weight']].to_csv(
        output_dir / 'network.csv', index=None)
    gt_df[['node_id', 'label']].to_csv(
        output_dir / 'gt.csv', index=None)
    np.save(arr=label_encoder.classes_,
            file=output_dir / 'label_encoder.npy')
    node_features_df.to_csv(
        output_dir / 'node_feature.csv', index=None)


if __name__ == '__main__':
    main()

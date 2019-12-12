import pandas as pd
import numpy as np
from src.data.utils import parse_data_path, get_dist
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def main(data_name):
    data_path_dict = parse_data_path(data_name)
    # 出力用フォルダ
    output_dir = Path('./data/processed/') / data_name
    if not output_dir.exists():
        output_dir.mkdir()
    # raw_network置き場
    raw_network_dir = Path('./data/raw/') / data_name
    if not raw_network_dir.exists():
        raw_network_dir.mkdir()
    # 処理
    file_path = data_path_dict['file_path']
    review_df_raw = pd.read_json(file_path, lines=True)
    assert ~review_df_raw.duplicated(['asin', 'reviewerID']).any()
    review_df = preprocessing(review_df_raw)
    network_df = generate_network_csv(review_df, pd.Timestamp(2013, 1, 1))
    network_df.columns = ['user_id', 'product_id', 'rating', 'time']
    network_df['weight'] = network_df.rating.map(lambda x: (x-3)/2).round()
    network_df.to_csv(raw_network_dir / 'network.csv')
    gt_df = generate_gt(review_df, pd.Timestamp(2013, 1, 1))
    gt_df.columns = ['user_id', 'label']
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
    pass


def preprocessing(df):
    df_ = df.copy()
    # converting 'reviewTime' column to datetime
    df_['reviewTime'] = pd.to_datetime(df_.reviewTime, format='%m %d, %Y')
    # vote_sum and helpful rate, helpful_bin
    df_['vote_sum'] = df_['helpful'].map(lambda x: x[1])
    df_['helpful_rate'] = df_['helpful'].map(
        lambda x: x[0]/x[1] if x[1] > 0 else float('nan'))
    df_['helpful_bin'] = pd.cut(df_.helpful_rate, bins=np.arange(
        0, 1.1, 0.1), include_lowest=True, labels=range(10))
    # おかしいデータを取り除く
    df_ = df_.loc[~(df_.helpful_rate > 1.0)]
    return df_


def generate_network_csv(df, from_date):
    review_df_from_ = df.loc[df.reviewTime >= from_date]
    return review_df_from_[['reviewerID', 'asin', 'overall', 'reviewTime']]


def generate_gt(df, from_date):
    reviewer_all_votes = \
        df.loc[df.reviewTime >= from_date].groupby('reviewerID', as_index=False)[
            'helpful'].agg(lambda x: list(np.vstack(x).sum(0)))

    reviewer_all_votes['vote_sum'] = reviewer_all_votes.helpful.map(
        lambda x: x[1])

    reviewer_all_votes['rate'] = reviewer_all_votes.helpful.map(
        lambda x: x[0]/x[1])
    selected_df = reviewer_all_votes.loc[(reviewer_all_votes.vote_sum >= 50) &
                                         ((reviewer_all_votes.rate <= 0.25) |
                                          (reviewer_all_votes.rate >= 0.75))]
    selected_df['label'] = selected_df['rate'].map(
        lambda x: -1 if x <= 0.25 else 1)

    return selected_df[['reviewerID', 'label']]


if __name__ == '__main__':
    main('amazon_baby')

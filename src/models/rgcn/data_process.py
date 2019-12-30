import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import torch


def get_dataset(data_name, edge_attribute='raw'):
    if 'amazon' in data_name:
        if edge_attribute == 'raw':
            dataset = get_amazon_dataset_rawedge(data_name)
        else:
            dataset = get_amazon_dataset_signedge(data_name)
    elif data_name in ['otc', 'alpha', 'epinions']:
        if edge_attribute == 'raw':
            dataset = get_others_dataset_rawedge(data_name)
        else:
            dataset = get_others_dataset_signedge(data_name)
    return dataset


def get_amazon_dataset_rawedge(data_name):
    data_dir = Path('./data/raw/') / data_name
    network_ = pd.read_csv(data_dir / 'network.csv', index_col=0)
    network_.columns = ['src_raw', 'dst_raw', 'etype', 'time', 'weight']
    network_inv = network_[['dst_raw', 'src_raw', 'etype', 'time']]
    network_inv.columns = ['src_raw', 'dst_raw', 'etype', 'time']
    network = network_.append(network_inv)

    etype_encoder = LabelEncoder()
    network['etype'] = etype_encoder.fit_transform(network.etype)
    gt = pd.read_csv(data_dir / 'gt.csv', index_col=0)

    gt.columns = ['node_id_raw', 'label']
    gt = gt.drop_duplicates('node_id_raw')

    # edge_normの計算
    amazon_src_cnt = network.groupby(['src_raw', 'etype'])[
        'time'].count().unstack(1, fill_value=0)

    amazon_src_dist = pd.DataFrame(
        amazon_src_cnt.values/amazon_src_cnt.sum(1).values.reshape(-1, 1),
        index=amazon_src_cnt.index,
        columns=amazon_src_cnt.columns)

    merged_network = pd.merge(
        network, amazon_src_dist.stack().reset_index(),
        on=['src_raw', 'etype'])

    label_encoder = LabelEncoder()
    label_encoder.fit(np.hstack((merged_network.src_raw,
                                 merged_network.dst_raw,
                                 gt.node_id_raw)))

    merged_network['src'] = label_encoder.transform(merged_network.src_raw)

    merged_network['dst'] = label_encoder.transform(merged_network.dst_raw)

    gt['node_id'] = label_encoder.transform(gt.node_id_raw)
    gt['label'] = gt['label'].map(lambda x: 1 if x == -1 else 0)

    # padding
    amazon_gt_padded = pd.merge(
        pd.DataFrame(np.arange(label_encoder.classes_.shape[0])), gt,
        left_index=True, right_on='node_id', how='left'
    ).fillna(0.5).sort_values('node_id')

    num_nodes = label_encoder.classes_.shape[0]
    num_rels = merged_network.etype.unique().shape[0]
    num_classes = gt.label.unique().shape[0]
    labels = amazon_gt_padded['label'].values.astype(int).reshape(-1, 1)
    all_idx = gt['node_id'].values

    # edge type and normalization factor
    edge_type = torch.from_numpy(merged_network['etype'].values)
    edge_norm = torch.from_numpy(
        merged_network[0].values.astype('float32')).unsqueeze(1)

    labels = torch.from_numpy(labels).view(-1)

    merged_network_directed = \
        merged_network.copy().iloc[:merged_network.shape[0]//2, :]
    node_feature_df = pd.concat(
        [get_dist(merged_network_directed, 'src'),
         get_dist(merged_network_directed, 'dst')], 1).fillna(0).sort_index()
    node_feature_array = node_feature_df.values.astype('float32')

    known_labels = gt['label'].values

    dataset = {
        "num_nodes": num_nodes,
        "num_rels": num_rels,
        "num_classes": num_classes,
        "labels": labels,
        "all_idx": all_idx,
        "edge_type": edge_type,
        "edge_norm": edge_norm,
        "labels": labels,
        "node_feature_array": node_feature_array,
        "known_labels": known_labels,
        "merged_network": merged_network,
    }
    return dataset


def get_amazon_dataset_signedge(data_name):
    data_dir = Path('./data/processed/') / data_name
    amazon_network_ = pd.read_csv(data_dir / 'network.csv')

    amazon_network_.columns = ['src_raw', 'dst_raw', 'etype']
    amazon_network_inv = amazon_network_[['dst_raw', 'src_raw', 'etype']]
    amazon_network_inv.columns = ['src_raw', 'dst_raw', 'etype']

    amazon_network = amazon_network_.append(amazon_network_inv)
    amazon_network['time'] = 1

    etype_encoder = LabelEncoder()
    amazon_network['etype'] = etype_encoder.fit_transform(amazon_network.etype)

    amazon_gt = pd.read_csv(data_dir / 'gt.csv')
    amazon_gt.columns = ['node_id_raw', 'label']
    amazon_gt = amazon_gt.drop_duplicates('node_id_raw')

    # edge_normの計算
    amazon_src_raw_cnt = amazon_network.groupby(['src_raw', 'etype'])[
        'dst_raw'].count().unstack(1, fill_value=0)

    amazon_src_raw_dist = pd.DataFrame(
        amazon_src_raw_cnt.values /
        amazon_src_raw_cnt.sum(1).values.reshape(-1, 1),
        index=amazon_src_raw_cnt.index,
        columns=amazon_src_raw_cnt.columns)

    merged_network = pd.merge(amazon_network, amazon_src_raw_dist.stack(
    ).reset_index(), on=['src_raw', 'etype'])

    label_encoder = LabelEncoder()
    label_encoder.fit(np.hstack((merged_network.src_raw,
                                 merged_network.dst_raw,
                                 amazon_gt.node_id_raw)))

    merged_network['src'] = label_encoder.transform(merged_network.src_raw)

    merged_network['dst'] = label_encoder.transform(merged_network.dst_raw)

    amazon_gt['node_id'] = label_encoder.transform(amazon_gt.node_id_raw)
    amazon_gt['label'] = amazon_gt['label'].map(lambda x: 1 if x == -1 else 0)

    # padding
    amazon_gt_padded = pd.merge(pd.DataFrame(np.arange(label_encoder.classes_.shape[0])), amazon_gt,
                                left_index=True, right_on='node_id', how='left').fillna(0.5).sort_values('node_id')

    num_nodes = label_encoder.classes_.shape[0]
    num_rels = merged_network.etype.unique().shape[0]
    num_classes = amazon_gt.label.unique().shape[0]
    labels = amazon_gt_padded['label'].values.astype(int).reshape(-1, 1)
    all_idx = amazon_gt['node_id'].values

    # edge type and normalization factor
    edge_type = torch.from_numpy(merged_network['etype'].values)
    edge_norm = torch.from_numpy(
        merged_network[0].values.astype('float32')).unsqueeze(1)

    labels = torch.from_numpy(labels).view(-1)

    # merged_network_directed = merged_network.copy().iloc[:merged_network.shape[0]//2,:]
    # node_feature_df = pd.concat([get_dist(merged_network_directed,'src'),get_dist(merged_network_directed,'dst')],1).fillna(0).sort_index()
    # node_feature_array = node_feature_df.values.astype('float32')
    node_feature_df = pd.read_csv(data_dir / 'node_feature.csv')
    node_feature_array = node_feature_df.values.astype('float32')
    known_labels = amazon_gt['label'].values

    dataset = {
        "num_nodes": num_nodes,
        "num_rels": num_rels,
        "num_classes": num_classes,
        "labels": labels,
        "all_idx": all_idx,
        "edge_type": edge_type,
        "edge_norm": edge_norm,
        "labels": labels,
        "node_feature_array": node_feature_array,
        "known_labels": known_labels,
        "merged_network": merged_network,
    }
    return dataset


def get_others_dataset_rawedge(data_name):
    data_dir = Path('./data/raw/') / data_name
    epinions_network = pd.read_csv(data_dir / 'network.csv', header=None,
                                   index_col=0)
    epinions_network.columns = ['src_raw',
                                'dst_raw', 'etype', 'time', 'weight']
    etype_encoder = LabelEncoder()
    epinions_network['etype'] = etype_encoder.fit_transform(
        epinions_network.etype)
    epinions_gt = pd.read_csv(data_dir / 'gt.csv', header=None, index_col=0)
    epinions_gt.columns = ['node_id_raw', 'label']
    epinions_gt = epinions_gt.drop_duplicates('node_id_raw')
    epinions_gt = epinions_gt.loc[epinions_gt.node_id_raw.isin(
        set(epinions_network.src_raw) | set(epinions_network.dst_raw))]

    # edge_normの計算
    epinions_src_cnt = epinions_network.groupby(['src_raw', 'etype'])[
        'time'].count().unstack(1, fill_value=0)

    epinions_src_dist = pd.DataFrame(epinions_src_cnt.values/epinions_src_cnt.sum(1).values.reshape(-1, 1),
                                     index=epinions_src_cnt.index,
                                     columns=epinions_src_cnt.columns)

    merged_network = pd.merge(epinions_network, epinions_src_dist.stack(
    ).reset_index(), on=['src_raw', 'etype'])

    label_encoder = LabelEncoder()
    label_encoder.fit(np.hstack((merged_network.src_raw,
                                 merged_network.dst_raw,
                                 epinions_gt.node_id_raw)))

    merged_network['src'] = label_encoder.transform(merged_network.src_raw)

    merged_network['dst'] = label_encoder.transform(merged_network.dst_raw)

    epinions_gt['node_id'] = label_encoder.transform(epinions_gt.node_id_raw)
    epinions_gt['label'] = epinions_gt['label'].map(
        lambda x: 1 if x == -1 else 0)

    # padding
    epinions_gt_padded = pd.merge(pd.DataFrame(np.arange(label_encoder.classes_.shape[0])), epinions_gt,
                                  left_index=True, right_on='node_id', how='left').fillna(0.5).sort_values('node_id')

    num_nodes = label_encoder.classes_.shape[0]
    num_rels = merged_network.etype.unique().shape[0]
    num_classes = epinions_gt.label.unique().shape[0]
    labels = epinions_gt_padded['label'].values.astype(int).reshape(-1, 1)
    all_idx = epinions_gt['node_id'].values

    # edge type and normalization factor
    edge_type = torch.from_numpy(merged_network['etype'].values)
    edge_norm = torch.from_numpy(
        merged_network[0].values.astype('float32')).unsqueeze(1)

    labels = torch.from_numpy(labels).view(-1)

    node_feature_df = pd.concat([get_dist(merged_network, 'src'), get_dist(
        merged_network, 'dst')], 1).fillna(0).sort_index()
    node_feature_array = node_feature_df.values.astype('float32')

    known_labels = epinions_gt['label'].values
    dataset = {
        "num_nodes": num_nodes,
        "num_rels": num_rels,
        "num_classes": num_classes,
        "labels": labels,
        "all_idx": all_idx,
        "edge_type": edge_type,
        "edge_norm": edge_norm,
        "labels": labels,
        "node_feature_array": node_feature_array,
        "known_labels": known_labels,
        "merged_network": merged_network,
    }
    return dataset


def get_others_dataset_signedge(data_name):
    data_dir = Path('./data/processed/') / data_name
    epinions_network = pd.read_csv(data_dir / 'network.csv')
    epinions_network.columns = ['src_raw', 'dst_raw', 'etype']
    etype_encoder = LabelEncoder()
    epinions_network['etype'] = etype_encoder.fit_transform(
        epinions_network.etype)
    epinions_gt = pd.read_csv(data_dir / 'gt.csv')
    epinions_gt.columns = ['node_id_raw', 'label']
    epinions_gt = epinions_gt.drop_duplicates('node_id_raw')
    epinions_network['time'] = 1

    # edge_normの計算
    epinions_src_cnt = epinions_network.groupby(['src_raw', 'etype'])[
        'time'].count().unstack(1, fill_value=0)

    epinions_src_dist = pd.DataFrame(
        epinions_src_cnt.values/epinions_src_cnt.sum(1).values.reshape(-1, 1),
        index=epinions_src_cnt.index,
        columns=epinions_src_cnt.columns)

    merged_network = pd.merge(epinions_network, epinions_src_dist.stack(
    ).reset_index(), on=['src_raw', 'etype'])

    label_encoder = LabelEncoder()
    label_encoder.fit(np.hstack((merged_network.src_raw,
                                 merged_network.dst_raw,
                                 epinions_gt.node_id_raw)))

    merged_network['src'] = label_encoder.transform(merged_network.src_raw)

    merged_network['dst'] = label_encoder.transform(merged_network.dst_raw)

    epinions_gt['node_id'] = label_encoder.transform(epinions_gt.node_id_raw)
    epinions_gt['label'] = epinions_gt['label'].map(
        lambda x: 1 if x == -1 else 0)

    # padding
    epinions_gt_padded = pd.merge(pd.DataFrame(np.arange(label_encoder.classes_.shape[0])), epinions_gt,
                                  left_index=True, right_on='node_id', how='left').fillna(0.5).sort_values('node_id')

    num_nodes = label_encoder.classes_.shape[0]
    num_rels = merged_network.etype.unique().shape[0]
    num_classes = epinions_gt.label.unique().shape[0]
    labels = epinions_gt_padded['label'].values.astype(int).reshape(-1, 1)
    all_idx = epinions_gt['node_id'].values

    # edge type and normalization factor
    edge_type = torch.from_numpy(merged_network['etype'].values)
    edge_norm = torch.from_numpy(
        merged_network[0].values.astype('float32')).unsqueeze(1)

    labels = torch.from_numpy(labels).view(-1)

    node_feature_df = pd.read_csv(
        data_dir / 'node_feature.csv')
    node_feature_array = node_feature_df.values.astype('float32')
    known_labels = epinions_gt['label'].values
    dataset = {
        "num_nodes": num_nodes,
        "num_rels": num_rels,
        "num_classes": num_classes,
        "labels": labels,
        "all_idx": all_idx,
        "edge_type": edge_type,
        "edge_norm": edge_norm,
        "labels": labels,
        "node_feature_array": node_feature_array,
        "known_labels": known_labels,
        "merged_network": merged_network,
    }
    return dataset


def get_dist(df, col):
    df_cnt = df.groupby(
        [col]+['etype'])['time'].count().unstack(1, fill_value=0)
    df_dist = pd.DataFrame(df_cnt.values / df_cnt.sum(1).values.reshape(-1, 1),
                           columns=df_cnt.columns,
                           index=df_cnt.index)
    return df_dist

import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.decomposition import TruncatedSVD


def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    dataset = pd.read_csv(args.edge_path).values.tolist()
    edges = {}
    edges["positive_edges"] = [edge[0:2] for edge in dataset if edge[2] == 1]
    edges["negative_edges"] = [edge[0:2] for edge in dataset if edge[2] == -1]
    edges["ecount"] = len(dataset)
    edges["ncount"] = len(
        set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))

    nodes_df = pd.read_csv(args.nodes_path)
    nodes_dict = {}
    nodes_dict['indice'] = nodes_df['node_id'].values
    nodes_dict['label'] = nodes_df['label'].values
    nodes_dict['all_ncount'] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset] +
                                       [node_id for node_id in nodes_df['node_id'].values]))
    return edges, nodes_dict


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] +
               [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def calculate_auc(targets, predictions, edges):
    """
    Calculate performance measures on test dataset.
    :param targets: Target vector to predict.
    :param predictions: Predictions vector.
    :param edges: Edges dictionary with number of edges etc.
    :return auc: AUC value.
    :return f1: F1-score.
    """
    # neg_ratio = len(edges["negative_edges"])/edges["ecount"]
    neg_ratio = (targets == 0).sum() / len(targets)
    # targets = [0 if target == 1 else 1 for target in targets]
    auc = roc_auc_score(targets, predictions)
    # f1 = f1_score(targets, [1 if p > neg_ratio else 0 for p in  predictions])
    return auc, 0


def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """
    t = Texttable()
    t.add_rows([per for i, per in enumerate(
        logs["performance"]) if i % 10 == 0])
    print(t.draw())


def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    with open(args.log_path, "w") as f:
        json.dump(logs, f)


def setup_features(args, positive_edges, negative_edges, node_count):
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    if args.spectral_features:
        X = create_spectral_features(
            args, positive_edges, negative_edges, node_count)
    else:
        X = create_general_features(args)
    return X


def create_general_features(args):
    """
    Reading features using the path.
    :param args: Arguments object.
    :return X: Node features.
    """
    X = np.array(pd.read_csv(args.features_path))
    return X


def create_spectral_features(args, positive_edges, negative_edges, node_count):
    """
    Creating spectral node features using the train dataset edges.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    print(node_count)
    p_edges = positive_edges + [[edge[1], edge[0]] for edge in positive_edges]
    n_edges = negative_edges + [[edge[1], edge[0]] for edge in negative_edges]
    train_edges = p_edges + n_edges
    index_1 = [edge[0] for edge in train_edges]
    index_2 = [edge[1] for edge in train_edges]
    values = [1]*len(p_edges) + [-1]*len(n_edges)
    shaping = (node_count, node_count)
    signed_A = sparse.csr_matrix(sparse.coo_matrix(
        (values, (index_1, index_2)), shape=shaping, dtype=np.float32))
    svd = TruncatedSVD(n_components=args.reduction_dimensions,
                       n_iter=args.reduction_iterations, random_state=args.seed)
    svd.fit(signed_A)
    X = svd.components_.T
    return X


def sample_edges(edges, sample_num, nodes_dict):
    edges_df = pd.DataFrame(edges['positive_edges'])
    edges_df = edges_df.append(pd.DataFrame(edges['negative_edges']))
    edges_df['weight'] = [1] * \
        len(edges['positive_edges']) + [-1]*len(edges['negative_edges'])
    first_neighbors = np.unique(edges_df.loc[(edges_df[0].isin(
        nodes_dict['indice'])) | (edges_df[1].isin(nodes_dict['indice']))].values)
    first_second = edges_df.loc[(edges_df[0].isin(first_neighbors)) | (
        edges_df[1].isin(first_neighbors))]
    sampled_positive_edges = first_second.loc[first_second.weight == 1, [
        0, 1]].values
    sampled_negative_edges = first_second.loc[first_second.weight == -1, [
        0, 1]].values
    positive_sample_num = min(sample_num, sampled_positive_edges.shape[0])
    sampled_positive_edges = sampled_positive_edges[np.random.choice(np.arange(
        sampled_positive_edges.shape[0]), positive_sample_num, replace=False), :]
    negative_sample_num = min(sample_num, sampled_negative_edges.shape[0])
    sampled_negative_edges = sampled_negative_edges[np.random.choice(np.arange(
        sampled_negative_edges.shape[0]), negative_sample_num, replace=False), :]
    return sampled_positive_edges, sampled_negative_edges

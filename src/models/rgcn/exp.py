from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from .data_process import get_dataset
import numpy as np
from dgl import DGLGraph
from .layers import Model
import torch
import torch.nn.functional as F


def ten_fold_cv(data_name):
    result_dict = {}
    raw_dataset = get_dataset(data_name, edge_attribute='raw')
    sign_dataset = get_dataset(data_name, edge_attribute='sign')
    # raw_dataset
    rawedge_auc = _ten_fold_for_dataset(**raw_dataset)
    # signed dataset
    signed_auc = _ten_fold_for_dataset(**sign_dataset)
    return {"raw_edge": rawedge_auc, "sign_edge": signed_auc}


def _ten_fold_for_dataset(all_idx, known_labels,
                          labels, num_nodes, merged_network,
                          edge_type, edge_norm,
                          num_classes, num_rels,
                          node_feature_array):
    # configurations
    n_hidden = [32, 16]  # number of hidden units
    n_bases = -1  # -1 # use number of relations as number of bases
    n_hidden_layers = 2  # use 1 input layer, 1 output layer, no hidden layer
    n_epochs = 50  # epochs to train
    lr = 0.025  # learning rate
    l2norm = 0.00001  # L2 norm coefficient
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    # 10-fold cross validation
    auc_scores = []

    for i, (for_train_val_idx, for_test_idx) in enumerate(
            kf.split(np.arange(len(all_idx)), y=known_labels)):
        train_val_idx = all_idx[for_train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.33,
            stratify=known_labels[for_train_val_idx])
        test_idx = all_idx[for_test_idx]
        if np.unique(labels[train_idx].detach().numpy()).shape[0] == 1 or np.unique(labels[val_idx].detach().numpy()).shape[0] == 1 \
                or np.unique(labels[test_idx].detach().numpy()).shape[0] == 1:
            print(np.unique(labels[test_idx].detach().numpy()))
            print('cont')
            continue
        # create graph
        g = DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(merged_network['src'].values, merged_network['dst'].values)
        g.edata.update({'rel_type': edge_type, 'norm': edge_norm})
        # create model
        model = Model(len(g),
                      n_hidden,
                      num_classes,
                      num_rels,
                      node_feature_array,
                      num_bases=n_bases,
                      num_hidden_layers=n_hidden_layers)
        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=l2norm)
        print("Start {}-th fold".format(i))
        print("==== Train Phase ====")
        model.train()
        best_auc = 0.0
        best_auc_logits = None
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            logits = model.forward(g)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()

            optimizer.step()

            train_auc = roc_auc_score(y_true=labels[train_idx].detach(
            ).numpy(), y_score=logits[train_idx].detach().numpy()[:, 1])
            train_loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            val_auc = roc_auc_score(y_true=labels[val_idx].detach(
            ).numpy(), y_score=logits[val_idx].detach().numpy()[:, 1])
            val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])

            if val_auc >= best_auc:
                best_auc = val_auc
                best_auc_logits = logits

            print("Epoch {:05d} | ".format(epoch) +
                  "Train AUC: {:.4f} | Train Loss: {:.4f} | ".format(
                train_auc, loss.item()) +
                "Validation AUC: {:.4f} | Validation loss: {:.4f}".format(
                val_auc, val_loss.item()))
        print("==== Test Phase ====")
        model.eval()
        test_auc = roc_auc_score(y_true=labels[test_idx].detach().numpy(
        ), y_score=best_auc_logits[test_idx].detach().numpy()[:, 1])
        auc_scores.append(test_auc)
        print("test auc : {}".format(test_auc))
        print("=================")
        return np.mean(auc_scores)


if __name__ == '__main__':
    ten_fold_cv('amazon_home')

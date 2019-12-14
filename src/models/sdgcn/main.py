from src.models.sdgcn.utils import read_graph, save_logs
from .args_getter import get_args
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from src.models.sdgcn.trainer import SignedGCNTrainer
import torch


def ten_fold_cv(data_name):
    args = get_args(data_name, rate=None)
    edges, nodes_dict = read_graph(args)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    all_indice = nodes_dict['indice']
    all_labels = nodes_dict['label']
    auc_scores = []
    regression_weights = []

    for i, (train_index, test_index) in enumerate(kf.split(X=nodes_dict['indice'], y=nodes_dict['label'])):
        print("==== Training Phase ====")
        print('{}-th fold'.format(i))
        # training
        train_node_indice = all_indice[train_index]
        train_node_labels = all_labels[train_index]
        print('labels:{}'.format(np.unique(train_node_labels, return_counts=True)))
        tmp_nodes_dict = {}
        tmp_nodes_dict['all_ncount'] = nodes_dict['all_ncount']
        tmp_nodes_dict['indice'] = train_node_indice
        tmp_nodes_dict['label'] = train_node_labels
        trainer = SignedGCNTrainer(args, edges, tmp_nodes_dict)
        trainer.setup_dataset()
        trainer.create_and_train_model()

        # if args.test_size > 0:
        #     save_logs(args, trainer.logs)

        # test
        print("==== Test Phase ====")
        test_node_indice = all_indice[test_index]
        test_node_labels = all_labels[test_index]
        # feature = pd.read_csv(args.embedding_path,index_col='id').values
        feature = pd.read_pickle(args.embedding_path).drop('id', 1).values
        test_feature = feature[test_node_indice]
        # weight = pd.read_csv(args.regression_weights_path)
        weight = pd.read_pickle(args.regression_weights_path)
        predictions = np.dot(test_feature, weight.values.T)
        probabilities = torch.sigmoid(torch.from_numpy(predictions)).numpy()
        predict_labels = (probabilities >= 0.5).astype(int)
        auc_score = roc_auc_score(
            y_true=[1 if i == -1 else 0 for i in test_node_labels], y_score=probabilities)
        auc_scores.append(auc_score)
        cmx = confusion_matrix(
            y_true=[1 if i == -1 else 0 for i in test_node_labels], y_pred=predict_labels)
        regression_weights.append(trainer.model.regression_weights)
        print("{0}-th fold's auc_score:{1}".format(i, auc_score))
        print(cmx)
        print()

    final_auc_score = np.mean(auc_scores)
    print("=======================================")
    print(f"averaged auc : {final_auc_score}")
    print("=======================================")
    return final_auc_score


if __name__ == '__main__':
    ten_fold_cv('otc')

from src.models.sgcn.utils import read_graph
from ..sdgcn.args_getter import get_args
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from src.models.sgcn.trainer import SignedGCNTrainer, SignedGCNPredictor
import torch
from pathlib import Path


def ten_fold_cv(experiment, data_name):
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
        experiment.log_metric(name=f'auc_fold_({i+1}/10)', value=auc_score)
        print("{0}-th fold's auc_score:{1}".format(i, auc_score))
        print(cmx)
        print()

    final_auc_score = np.mean(auc_scores)
    experiment.log_metric("average_auc_10fold", final_auc_score)
    print("=======================================")
    print(f"averaged auc : {final_auc_score}")
    print("=======================================")
    return {"auc": final_auc_score}


def robustness_experiments(
        experiment,
        data_name,
        training_rates_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        iter_num=30):

    args = get_args(data_name, rate=None)
    edges, nodes_dict = read_graph(args)
    all_indice = nodes_dict['indice']
    all_labels = nodes_dict['label']

    all_auc_scores = []
    done_train_rate = []
    training_rates = training_rates_list
    for train_rate in training_rates:
        print('train_rate : ', train_rate)
        auc_scores = []
        for i in range(iter_num):
            train_index, test_index = train_test_split(np.arange(len(nodes_dict['indice'])),
                                                       stratify=nodes_dict['label'], train_size=float(train_rate), shuffle=True)
            print("==== Training Phase ====")
            print('{}-th'.format(i))
            # training
            train_node_indice = all_indice[train_index]
            train_node_labels = all_labels[train_index]
            print('labels:{}'.format(
                np.unique(train_node_labels, return_counts=True)))
            tmp_nodes_dict = {}
            tmp_nodes_dict['all_ncount'] = nodes_dict['all_ncount']
            tmp_nodes_dict['indice'] = train_node_indice
            tmp_nodes_dict['label'] = train_node_labels
            trainer = SignedGCNTrainer(args, edges, tmp_nodes_dict)
            trainer.setup_dataset()
            trainer.create_and_train_model()

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
            probabilities = torch.sigmoid(
                torch.from_numpy(predictions)).numpy()
            predict_labels = (probabilities >= 0.5).astype(int)
            auc_score = roc_auc_score(
                y_true=[1 if i == -1 else 0 for i in test_node_labels], y_score=probabilities)
            auc_scores.append(auc_score)
            cmx = confusion_matrix(
                y_true=[1 if i == -1 else 0 for i in test_node_labels], y_pred=predict_labels)
            print("{0}-th fold's auc_score:{1}".format(i, auc_score))
            print(cmx)
            print()

        all_auc_scores.append(np.mean(auc_scores))
        done_train_rate.append(train_rate)
        print(train_rate, ':', np.mean(auc_scores))
        experiment.log_metric(f'rate_{train_rate}', np.mean(auc_scores))
    result_df = pd.DataFrame(
        all_auc_scores, index=done_train_rate, columns=['average_auc'])
    return result_df


def inductive_learning_eval(
        exp4_select,
        experiment, data_name, rate_list=[0.1, 0.2, 0.3], iter_num=30):
    if 'train' in exp4_select:
        _train_all(experiment, data_name, rate_list, iter_num=iter_num)
    if 'eval' in exp4_select:
        new_args = get_args(data_name)
        result_df, true_pred_dict = _eval_all(
            experiment, data_name, rate_list, new_args, iter_num=iter_num)
    else:
        result_df = pd.DataFrame()
    return result_df


def _train_all(experiment, data_name, rate_list,
               l1_lambda=0.0, l2_lambda=10e-4, iter_num=30):
    inductive_model_dir = Path('./models/sgcn/')
    inductive_model_dir.mkdir(parents=True, exist_ok=True)
    trained_rate = []
    for rate in rate_list:
        for i in range(iter_num):
            print(f'{i}-th iteration')
            inductive_model_path = inductive_model_dir / \
                f'{data_name}_{rate}_{i}th'
            args = get_args(data_name, rate=rate,
                            inductive_model_path=inductive_model_path,
                            l1_lambda=1.0)
            edges, nodes_dict = read_graph(args)
            print(np.unique(nodes_dict['label'], return_counts=True))
            trainer = SignedGCNTrainer(args, edges, nodes_dict)
            trainer.setup_dataset()
            trainer.create_and_train_model()
        trained_rate.append(rate)
        experiment.log_other('inductive_trained_rate', str(trained_rate))


def _eval_all(experiment, data_name, rate_list, new_args, iter_num=30):
    new_edges, new_nodes_dict = read_graph(new_args)
    X = np.array(pd.read_csv(new_args.features_path))
    result_df = pd.DataFrame()
    true_pred_dict = {}
    for rate in rate_list:
        auc_scores = []
        bagging_pred_scores = []
        for i in range(iter_num):
            inductive_model_path = Path(
                './models/sgcn/') / f'{data_name}_{rate}_{i}th'
            predictor = SignedGCNPredictor(
                new_args,
                inductive_model_path,
                X, new_edges, new_nodes_dict)
            predictions = predictor.predict()
            trained_node_raw = np.load(
                f'./data/processed/early/{data_name}_{rate}/label_encoder.npy')
            newly_added_node_judger = ~np.isin(
                new_nodes_dict['indice'], trained_node_raw)
            y_true = new_nodes_dict['label'][newly_added_node_judger]
            y_score_indice = new_nodes_dict['indice'][newly_added_node_judger]
            bagging_pred_scores.append(predictions[y_score_indice])

            current_auc = roc_auc_score(
                y_true=[1 if i == -1 else 0 for i in y_true], y_score=predictions[y_score_indice])
            auc_scores.append(current_auc)
        averaged_auc = np.mean(auc_scores)
        result_df = result_df.append(pd.io.json.json_normalize(
            {f'{data_name}_{rate}': averaged_auc}).T)
        experiment.log_other(key=f'exp4_rate_{rate}', value=averaged_auc)

        bagging_pred_score = sum(bagging_pred_scores)/iter_num
        tmp_df = pd.DataFrame()
        tmp_df['indice'] = y_score_indice
        tmp_df['true_label'] = [1 if i == -1 else 0 for i in y_true]
        tmp_df['pred_score'] = bagging_pred_score
        true_pred_dict[(data_name, rate)] = tmp_df
    return result_df, true_pred_dict

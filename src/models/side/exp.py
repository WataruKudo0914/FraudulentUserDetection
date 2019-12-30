from pathlib import Path
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import pandas as pd


def _load_data(data_name):
    side_output_dir = Path('./data/intermediate/side')
    emb1_df = pd.read_csv(
        side_output_dir / f'{data_name}.emb', sep=' ', header=None)
    emb2_df = pd.read_csv(
        side_output_dir / f'{data_name}.emb2', sep=' ', header=None)
    emb_df = pd.concat([emb1_df, emb2_df], 1)
    bias_df = pd.read_csv(
        side_output_dir / f'{data_name}.bias', sep=' ', header=None)
    vocab_df = pd.read_csv(
        side_output_dir / f'{data_name}.vocab', sep=' ', header=None, usecols=[0, 1])
    vocab_df['original_id'] = vocab_df[0].str.extract(
        ".*'(\d*).*'").astype(int)
    gt_df = pd.read_csv(
        f'./data/processed/{data_name}/gt.csv').drop_duplicates('node_id')
    vocab_gt_df = pd.merge(
        vocab_df, gt_df, left_on='original_id', right_on='node_id', how='left')
    X = pd.concat([emb_df, bias_df], 1).values[vocab_gt_df.label.notnull()]
    y = vocab_gt_df.label.map(lambda x: 1 if x == -
                              1 else 0).values[vocab_gt_df.label.notnull()]
    return X, y


def ten_fold_cv(experiment, data_name):
    X, Y = _load_data(data_name)
    X, Y = shuffle(X, Y)
    skf = StratifiedKFold(n_splits=10)
    scores = []
    aucscores = []
    i = 0
    for train, test in skf.split(X, Y):
        i += 1
        train_X = X[train]
        train_Y = Y[train]
        test_X = X[test]
        test_Y = Y[test]

        clf = RandomForestClassifier(n_estimators=500)
        clf.fit(train_X, train_Y)
        scores.append(accuracy_score(test_Y, clf.predict(test_X)))
        pred_Y = clf.predict_proba(test_X)
        false_positive_rate, true_positive_rate, th = roc_curve(
            test_Y, pred_Y[:, 1])
        auc_score = auc(false_positive_rate, true_positive_rate)
        aucscores.append(auc_score)
        experiment.log_metric(name=f'auc_fold_({i}/10)', value=auc_score)
        print(scores[-1], aucscores[-1])

    final_auc_score = np.mean(aucscores)
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
    X, Y = _load_data(data_name)
    X, Y = shuffle(X, Y)

    all_auc_scores = []
    done_train_rate = []
    for train_rate in training_rates_list:
        print('train_rate : ', train_rate)
        scores = []
        auc_scores = []
        for i in range(iter_num):
            train_X, test_X, train_Y, test_Y = \
                train_test_split(
                    X, Y, train_size=float(train_rate),
                    shuffle=True, stratify=Y)
            clf = RandomForestClassifier(n_estimators=500)
            clf.fit(train_X, train_Y)
            scores.append(accuracy_score(test_Y, clf.predict(test_X)))
            pred_Y = clf.predict_proba(test_X)
            false_positive_rate, true_positive_rate, th = roc_curve(
                test_Y, pred_Y[:, 1])
            auc_score = auc(false_positive_rate, true_positive_rate)
            auc_scores.append(auc_score)

        all_auc_scores.append(np.mean(auc_scores))
        done_train_rate.append(train_rate)
        print(train_rate, ':', np.mean(auc_scores))
        experiment.log_metric(f'rate_{train_rate}', np.mean(auc_scores))
    result_df = pd.DataFrame(
        all_auc_scores, index=done_train_rate, columns=['average_auc'])
    return result_df

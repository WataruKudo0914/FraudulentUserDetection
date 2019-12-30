from pathlib import Path
from collections import defaultdict
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import pandas as pd


def _load_data(data_name):
    rev2_result_path = Path(
        f"./data/intermediate/rev2_results/fairness/{data_name}")
    scores = defaultdict(list)
    fnames = os.listdir(rev2_result_path)
    for fname in fnames:
        f = open(rev2_result_path / fname, "r")
        for l in f:
            l = l.strip().split(",")
            if l[1] == "nan":
                l[1] = "0"
            scores[l[0]].append(float(l[1]))

    # create score vectors for ground truth nodes
    f = open("./data/raw/%s/gt.csv" % data_name, "r")
    X = []
    Y = []

    for l in f:
        l = l.strip().split(",")[1:]
        d = scores['u' + l[0]]
        if d == []:
            continue
        if l[1] == "-1":
            # badusers.add('u'+l[0])
            Y.append(1)
            X.append(scores['u'+l[0]])
        else:
            # goodusers.add('u'+l[0])
            Y.append(0)
            X.append(scores['u'+l[0]])
    f.close()
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


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

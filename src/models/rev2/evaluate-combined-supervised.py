# This code calculates the mean average precision score, for all input parameter combinations, using all outputs of the rev2 algorithm.
# This is for the supervised case.

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc
from collections import defaultdict
import numpy
import os
from pathlib import Path

network = 'amazon_app'  # sys.argv[1]
rev2_result_path = Path(f"/home2/kudo/FraudulentUserDetection/data/"
                        f"intermediate/rev2_results/fairness/{network}")

# load stored scores from rev2 runs with different parameter combinations
scores = defaultdict(list)
fnames = os.listdir(rev2_result_path)
for fname in fnames:
    #     if network not in fname:
    #         continue
    #     if "result" in fname:
    #         continue
    f = open(rev2_result_path / fname, "r")
    for l in f:
        l = l.strip().split(",")
        if l[1] == "nan":
            l[1] = "0"
        scores[l[0]].append(float(l[1]))
        # if l[2] == "nan":
        #     l[2] = "0"
        # scores[l[0]].append(float(l[2]))

# create score vectors for ground truth nodes
f = open("./data/raw/%s/gt.csv" % network, "r")
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

print(len(X), len(Y))

# train random forest classifier

X = numpy.array(X)
Y = numpy.array(Y)

X, Y = shuffle(X, Y)
skf = StratifiedKFold(Y, 10)
scores = []
aucscores = []
for train, test in skf:
    train_X = X[train]
    train_Y = Y[train]
    test_X = X[test]
    test_Y = Y[test]

    clf = RandomForestClassifier(n_estimators=500)
    clf.fit(train_X, train_Y)
    scores.append(accuracy_score(test_Y, clf.predict(test_X)))
    try:
        pred_Y = clf.predict_proba(test_X)
        false_positive_rate, true_positive_rate, th = roc_curve(
            test_Y, pred_Y[:, 1])
        aucscores.append(auc(false_positive_rate, true_positive_rate))
    except:
        pass
    print(scores[-1], aucscores[-1])

print("Accuracy scores", scores, numpy.mean(scores))
print("AUC scores", aucscores, numpy.mean(aucscores))

import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def get_acc_classifier(x, y, classifier="linear"):
    if classifier=="linear":
        clf = LogisticRegression()
    elif classifier=="tree":
        clf = DecisionTreeClassifier(max_depth=5)
    elif classifier=="random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    elif classifier=="gb":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=5)
    elif classifier=="mlp":
        clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)


def calc_accs(results, n_samples=None):
    if n_samples is None:
        n_samples = results['y'].shape[0]
    idxs = random.choices(range(results['y'].shape[0]), k=n_samples)
    accs = np.zeros((results['y'].shape[1], results['z'].shape[1]))
    accs_all = np.zeros(results['y'].shape[1])
    for i in range(results['y'].shape[1]):
        classes = list(set(results['y'][:,i].numpy()))
        y_i = np.array([classes.index(i) for i in results['y'][idxs,i][:,None].numpy()])
        accs_all[i] = get_acc_classifier(results['z'][idxs].numpy(), y_i, "random_forest")
        for j in range(results['z'].shape[1]):
            accs[i,j] = get_acc_classifier(results['z'][idxs,j][:,None].numpy(), y_i, "random_forest")
    return {
        "accs_all_{}".format(n_samples):        accs_all.mean(),
        "accs_{}".format(n_samples):            accs.max(axis=1).mean(),
    }
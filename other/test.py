import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

data_path = 'D:\\resource\\project\\test\\'
# data_path = '/home/liziyang/set/'

file1 = "train_hungarian_2.csv"
file2 = "train_long-beach-va_2.csv"
file3 = "train_switzerland_2.csv"

set1 = np.loadtxt(open(data_path + file1, "rb"), delimiter=',', skiprows=1, comments='#')
set2 = np.loadtxt(open(data_path + file2, "rb"), delimiter=',', skiprows=1, comments='#')


def combine(set1, set2, type):
    if type == 0:
        return {"set": np.vstack((set1, set2)).tolist()}
    else:
        return {"set": np.hstack((set1, set2)).tolist()}


def T(set):
    set = np.array(set)
    return {"set": set.T.tolist()}


def transpose(set):
    set = np.array(set)
    return {"set": set.transpose().tolist()}


def swapaxes(set, axis1, axis2):
    set = np.array(set)
    return {"set": set.swapaxes(axis1, axis2).tolist()}


def cut(set, row, column):
    set = np.array(set)
    r = strCut(row)
    c = strCut(column)
    if len(r) == 1 and len(c) == 1:
        set = set[r[0], c[0]]
    elif len(r) == 2 and len(c) == 1:
        if r[0] == None and r[1] == None:
            set = set[:, c[0]]
        elif r[0] != None and r[1] == None:
            set = set[r[0]:, c[0]]
        elif r[0] == None and r[1] != None:
            set = set[:r[1], c[0]]
        else:
            set = set[r[0]:r[1], c[0]]
    elif len(r) == 1 and len(c) == 2:
        if c[0] == None and c[1] == None:
            set = set[r[0], :]
        elif c[0] != None and c[1] == None:
            set = set[r[0], c[0]:]
        elif c[0] == None and c[1] != None:
            set = set[r[0], :c[1]]
        else:
            set = set[r[0], c[0]:c[1]]
    else:
        if r[0] == None and r[1] == None:
            set = set[:, :]
        elif r[0] != None and r[1] == None:
            set = set[r[0]:, :]
        elif r[0] == None and r[1] != None:
            set = set[:r[1], :]
        else:
            set = set[r[0]:r[1], :]

        if c[0] == None and c[1] == None:
            set = set[:, :]
        elif c[0] != None and c[1] == None:
            set = set[:, c[0]:]
        elif c[0] == None and c[1] != None:
            set = set[:, :c[1]]
        else:
            set = set[:, c[0]:c[1]]
    return {"set": set.tolist()}


def strToInt(str):
    try:
        return int(str)
    except ValueError:
        return None


def strCut(str):
    aaa = str.split(":")
    back = []
    if len(aaa) == 1:
        num = strToInt(aaa[0])
        if num != None:
            back.append(num)
        else:
            raise Exception('The input "' + str + '" is not an integer!\n' +
                            'The format for example is "1" ":" "1:" ":1" "1:2"')
    elif len(aaa) == 2:
        if len(aaa[0]) > 0:
            start = strToInt(aaa[0])
            if start != None:
                back.append(start)
            else:
                raise Exception('The input "' + str + '" has a incorrect format!\n' +
                                'The format for example is "1" ":" "1:" ":1" "1:2"')
        else:
            back.append(None)
        if len(aaa[1]) > 0:
            end = strToInt(aaa[1])
            if end != None:
                back.append(end)
            else:
                raise Exception('The input "' + str + '" has a incorrect format!\n' +
                                'The format for example is "1" ":" "1:" ":1" "1:2"')
        else:
            back.append(None)
    else:
        raise Exception('The input "' + str + '" has a incorrect format!\n' +
                        'The format for example is "1" ":" "1:" ":1" "1:2"')
    return back


def shuffle1(set):
    set = np.array(set)
    set = shuffle(set)
    return {"set": set.tolist()}


def svm(x_train, x_test, y_train, y_test,
        C, kernel, degree, gamma, coef0, shrinking, probability,
        tol, cache_size, class_weight, verbose,
        max_iter, decision_function_shape, break_ties, random_state):
    clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
              coef0=coef0, shrinking=shrinking, probability=probability,
              tol=tol, cache_size=cache_size, class_weight=class_weight,
              verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
              break_ties=break_ties,
              random_state=random_state)

    clf.fit(x_train, y_train)

    return {'score': clf.score(x_test, y_test), 'predict': clf.predict(x_test)}


def knn(x_train, x_test, y_train, y_test,
        n_neighbors, weights, algorithm,
        leaf_size, p, metric, metric_params, n_jobs):
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size,
        p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)

    clf.fit(x_train, y_train)

    return {'score': clf.score(x_test, y_test), 'predict': clf.predict(x_test)}


def dt(x_train, x_test, y_train, y_test,
       criterion,
       splitter,
       max_depth,
       min_samples_split,
       min_samples_leaf,
       min_weight_fraction_leaf,
       max_features,
       random_state,
       max_leaf_nodes,
       min_impurity_decrease,
       min_impurity_split,
       class_weight,
       presort,
       ccp_alpha):
    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        random_state=random_state,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        min_impurity_split=min_impurity_split,
        class_weight=class_weight,
        presort=presort,
        ccp_alpha=ccp_alpha)

    clf.fit(x_train, y_train)
    return {'score': clf.score(x_test, y_test), 'predict': clf.predict(x_test)}


def rf(x_train, x_test, y_train, y_test,
       n_estimators,
       criterion,
       max_depth,
       min_samples_split,
       min_samples_leaf,
       min_weight_fraction_leaf,
       max_features,
       max_leaf_nodes,
       min_impurity_decrease,
       min_impurity_split,
       bootstrap,
       oob_score,
       n_jobs,
       random_state,
       verbose,
       warm_start,
       class_weight,
       ccp_alpha,
       max_samples
       ):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        min_impurity_split=min_impurity_split,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        warm_start=warm_start,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        max_samples=max_samples)

    clf.fit(x_train, y_train)

    return {'score': clf.score(x_test, y_test), 'predict': clf.predict(x_test)}


def run(*arrays, test_size, train_size, random_state, shuffle):
    set = train_test_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state,
                           shuffle=shuffle)
    return {"set": set}


def main(*arrays, test_size=None, train_size=None, random_state=None, shuffle=None):
    return run(*arrays, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle)


set = train_test_split(set1)

x_1 = set1[:, :-1]
y_1 = set1[:, - 1]

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_1, y_1, test_size=0.2, random_state=1)

clf = SVC()

clf.fit(x_train1, y_train1)
print({'score': clf.score(x_test1, y_test1), 'predict': clf.predict(x_test1).tolist()})

print(main({'as': 2}, 1))
def main(array, index):
    return {"value": array[index]}


print(main({'as': 2}, 1))

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# 归一化 min_max
def min_max(x_train, x_test):
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)
    return x_train, x_test


def min_max_one(x):
    min_max_scaler = MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x


# 标准化 Z_score
def Z_score(x_train, x_test):
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


# 标准化 Z_score
def Z_score_one(x):
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    return x


# 特征选择
def FeatureSelection(x):
    selector = VarianceThreshold()
    selector.fit(x)
    return selector.get_support(True)


def FeaturePipeLine(estimator, x_train, y_train, x_test, y_test):
    anova_filter = SelectKBest(f_regression, k=2)
    poly = PolynomialFeatures(3, include_bias=False)
    anova_svm = Pipeline([('anova', anova_filter), ('poly', poly), ('estimator', estimator)])
    anova_svm.fit(x_train, y_train)
    return anova_svm.score(x_test, y_test)


def SplitSet(set):
    return set[:, range(len(set[0, :]) - 1)], set[:, -1]


def ProcessSetOne(x):
    columns_count = len(x[0, :]) - 1
    x = Z_score_one(x)
    x_columns = FeatureSelection(x[:, range(columns_count)])
    return x[:, x_columns]


def ProcessSet(x_train, x_test):
    columns_count = len(x_train[0, :]) - 1
    x_train, x_test = Z_score(x_train, x_test)
    x_columns = FeatureSelection(x_train[:, range(columns_count)])
    return x_train[:, x_columns], x_test[:, x_columns]


def ToOne(set):
    columns_count = len(set[0, :]) - 1
    rows_count = len(set)

    for i in range(columns_count):
        max = np.max(set[:, i])
        for j in range(rows_count):
            set[j, i] = set[j, i] / max

    return set


def SignalProcess(set, columes):
    columns_count = len(set[0, :]) - 1
    set_back = []

    for i in [0, 1, 2, 3, 4]:
        set1 = set[set[:, -1] == i]
        for j in range(columns_count):
            set2 = set1[:, j]
            if j in columes:
                set2 = signal.medfilt(set2, 9)
            if j == 0:
                set_mid = set2.reshape(set2.shape[0], 1)
            else:
                set_mid = np.hstack((set_mid, set2.reshape(set2.shape[0], 1)))

        label = i * np.ones(len(set1))
        set_mid = np.hstack((set_mid, label.reshape(label.shape[0], 1)))

        if i == 0:
            set_back = set_mid
        else:
            set_back = np.vstack((set_back, set_mid))

    return set_back


def run(filePath):
    set = np.loadtxt(open(filePath, "rb"), delimiter=',', skiprows=1, comments='#')

    columns_count = len(set[0, :])

    set = SignalProcess(set, range(6))

    set = ToOne(set)

    set = shuffle(set)

    x_ = set[:, range(columns_count - 1)]
    y_ = set[:, columns_count - 1]

    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=1)

    return sk(x_train, y_train, x_test, y_test)


# 计算分类算法的得分情况
def sk(x_train, y_train, x_test, y_test):
    clf1 = KNeighborsClassifier(n_neighbors=2)
    clf2 = DecisionTreeClassifier(max_depth=8)
    clf3 = RandomForestClassifier(n_estimators=14, max_depth=5)
    clf4 = SVC(C=100000, kernel='rbf', gamma='auto', probability=True)

    clf1.fit(x_train, y_train)
    clf2.fit(x_train, y_train)
    clf3.fit(x_train, y_train)
    clf4.fit(x_train, y_train)

    score1 = clf1.score(x_test, y_test)
    score2 = clf2.score(x_test, y_test)
    score3 = clf3.score(x_test, y_test)
    score4 = clf4.score(x_test, y_test)
    predict1 = clf1.predict(x_test).tolist()
    predict2 = clf2.predict(x_test).tolist()
    predict3 = clf3.predict(x_test).tolist()
    predict4 = clf4.predict(x_test).tolist()
    x = []
    for i in range(len(predict1)):
        x.append(i)
    return {
        'score': {'data': [score1, score2, score3, score4], 'x': ['KNeighbors', 'DecisionTree', 'RandomForest', 'SVM']},
        'predict': {'data': [predict1, predict2, predict3, predict4], 'x': x,
                    'y': ['KNeighbors', 'DecisionTree', 'RandomForest', 'SVM']}}


print(run("D:\\resource\\project\\test\\train_all_3.csv"))

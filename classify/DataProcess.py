from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import scipy.signal as signal
# from statsmodels.robust import mad
import matplotlib.pyplot as plt
import math


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
                # set2 = wavelet_denoising(set2)
                # print(np.mean(set2))
                # set2 = pywt.wavedec(set2, 2, 'soft')
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


# def wavelet_denoising(data):
#     sign = False
#     row_count = len(data)
#     if row_count % 2 == 1:
#         data = np.append(data, np.mean(data))
#         sign = True
#
#     (cA, cD) = pywt.dwt(data, "db4")
#     thresh1 = mad(cA) * math.sqrt(2 * math.log(row_count))  # 细节系数阈值（固定阈值）
#     thresh2 = mad(cD) * math.sqrt(2 * math.log(row_count))  # 近似系数阈值（固定阈值）
#     cAt = pywt.threshold(cA, thresh1, mode="soft")
#     cDt = pywt.threshold(cD, thresh2, mode="soft")
#     meta = pywt.idwt(cAt, cDt, "db4")
#     if sign:
#         meta = np.delete(meta, -1)
#     return meta

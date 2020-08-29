from classify.DataProcess import *
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt


# 绘制learning_curve学习曲线：展示不同数据量，算法学习得分
def plot_learning_curve(estimator, title, x, y, cv, n_jobs):
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test score")  # Cross-validation

    plt.legend(loc="best")
    return plt


# 绘制validation_curve验证曲线：展示某个因子，不同取值的算法得分
def plot_validation_curve(estimator, title, x, y, cv, param_name, param_range):
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_score, validation_score = validation_curve(estimator, x, y, param_name=param_name, cv=cv, param_range=param_range)

    train_scores_mean = np.mean(train_score, axis=1)
    train_scores_std = np.std(train_score, axis=1)
    validation_scores_mean = np.mean(validation_score, axis=1)
    validation_scores_std = np.std(validation_score, axis=1)

    plt.grid()

    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_std + train_scores_std, color='r', alpha=0.1)
    plt.fill_between(param_range, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, color='g', alpha=0.1)

    plt.plot(param_range, train_scores_mean, 'o-', c='r', label='train score')
    plt.plot(param_range, validation_scores_std, 'o-', c='g', label='validation score')

    plt.legend(loc='best')
    return plt


# 输出学习曲线
def ShowLearnLine(estimator, title, set, cv=3, n_jobs=1):
    x, y = SplitSet(set)
    x = ProcessSetOne(x)

    plt.figure('learning_curve_' + title)
    plot_learning_curve(estimator, title, x, y, cv=cv, n_jobs=n_jobs)
    return plt


# 显示验证曲线
def ShowValidationLine(estimator, title, set, param_name, param_range, cv=3):
    x, y = SplitSet(set)
    x = ProcessSetOne(x)

    plt.figure('Validation_Line_' + title)
    plot_validation_curve(estimator, title, x, y, cv=cv, param_name=param_name, param_range=param_range)
    return plt

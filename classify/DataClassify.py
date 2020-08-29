from classify.set import *
from classify.DataProcess import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

score1_1, score2_1, score3_1, score4_1, score5_1 = -1, -1, -1, -1, -1
score1_2, score2_2, score3_2, score4_2, score5_2 = -1, -1, -1, -1, -1
score1_3, score2_3, score3_3, score4_3, score5_3 = -1, -1, -1, -1, -1
score1_4, score2_4, score3_4, score4_4, score5_4 = -1, -1, -1, -1, -1


# 计算分类算法的得分情况
def sk(x_train, y_train, x_test, y_test):
    score1, score2, score3, score4, score5 = -1, -1, -1, -1, -1

    clf1 = KNeighborsClassifier(n_neighbors=2)
    clf2 = DecisionTreeClassifier(max_depth=8)
    clf3 = RandomForestClassifier(n_estimators=14, max_depth=5)
    clf4 = SVC(C=100000,kernel='rbf', gamma='auto', probability=True)
    # clf5 = VotingClassifier(estimators=[('knn', clf1), ('dt', clf2), ('rf', clf3), ('svc', clf4)], voting='soft', weights=[1, 1, 1,1])

    clf1.fit(x_train, y_train)
    clf2.fit(x_train, y_train)
    clf3.fit(x_train, y_train)
    clf4.fit(x_train, y_train)

    score1 = clf1.score(x_test, y_test)
    score2 = clf2.score(x_test, y_test)
    score3 = clf3.score(x_test, y_test)
    score4 = clf4.score(x_test, y_test)
    print(clf3.predict(x_test))
    print(y_test)
    # print(y_test)
    # print(clf4.predict(x_test))
    # score5 = clf5.score(x_test, y_test)

    return score1, score2, score3, score4


# def FindBestParameters(x, y, name, cv=3, n_jobs=4):
#     x = Z_score_one(x)
#     clf1 = KNeighborsClassifier()
#     clf2 = DecisionTreeClassifier()
#     clf3 = RandomForestClassifier()
#     clf4 = SVC(gamma='auto')
#
#     # parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': range(1, 10), 'degree': range(1, 5), 'gamma': ['auto'],
#     #               'shrinking': ['true','false'], 'probability': ['true','false']}
#     parameters1 = {'n_neighbors': range(1, 30), }
#     # parameters2 = {'max_depth': range(1, len(x)), 'min_samples_leaf': range(1, 40), 'min_samples_split': range(2, 40)}
#     # parameters3 = {'n_estimators': range(1, len(x)), 'max_depth': range(1, len(x)), 'min_samples_leaf': range(1, 40), 'min_samples_split': range(2, 40)}
#     parameters2 = {'max_depth': range(1, len(x))}
#     parameters3 = {'n_estimators': range(1, len(x)), 'max_depth': range(1, 30)}
#     parameters4 = {'kernel': ('linear', 'rbf', 'sigmoid'), 'C': range(1, 10)}
#
#     gs1 = GridSearchCV(clf1, parameters1, cv=cv, n_jobs=n_jobs)
#     # gs1.fit(x, y)
#     # pd.DataFrame(data=gs1.cv_results_).to_csv(data_path + 'result_' + name + '_KNN' + '.csv')
#     # print('---> result_' + name + '_KNN' + '.csv is done!')
#     # gs2 = GridSearchCV(clf2, parameters2, cv=cv, n_jobs=n_jobs)
#     # gs2.fit(x, y)
#     # pd.DataFrame(data=gs2.cv_results_).to_csv(data_path + 'result_' + name + '_DT' + '.csv')
#     # print('---> result_' + name + '_DT' + '.csv is done!')
#     gs3 = GridSearchCV(clf3, parameters3, cv=cv, n_jobs=n_jobs)
#     gs3.fit(x, y)
#     pd.DataFrame(data=gs3.cv_results_).to_csv(data_path + 'result_' + name + '_RF' + '.csv')
#     print('---> result_' + name + '_RF' + '.csv is done!')
#     # gs4 = GridSearchCV(clf4, parameters4, cv=cv, n_jobs=n_jobs)
#     # gs4.fit(x, y)
#     # pd.DataFrame(data=gs4.cv_results_).to_csv(data_path + 'result_' + name + '_SVC' + '.csv')
#     # print('---> result_' + name + '_SVC' + '.csv is done!')
#
#
# def SetCount():
#     print(np.sum(set1[:, -1] == 0), np.sum(set1[:, -1] == 1), np.sum(set1[:, -1] == 2), np.sum(set1[:, -1] == 3), np.sum(set1[:, -1] == 4))
#     # print(np.sum(set2[:, -1] == 0), np.sum(set2[:, -1] == 1), np.sum(set2[:, -1] == 2), np.sum(set1[:, -1] == 3), np.sum(set2[:, -1] == 4))
#     # print(np.sum(set3[:, -1] == 0), np.sum(set3[:, -1] == 1), np.sum(set3[:, -1] == 2), np.sum(set1[:, -1] == 3), np.sum(set3[:, -1] == 4))
#     # print(np.sum(set4[:, -1] == 0), np.sum(set4[:, -1] == 1), np.sum(set4[:, -1] == 2), np.sum(set1[:, -1] == 3), np.sum(set4[:, -1] == 4))

if __name__ == '__main__':
    # SetCount()

    # sk函数调用，打印各个算法的分数
    score1_1, score2_1, score3_1, score4_1 = sk(x_train4, y_train4, x_test4, y_test4)
    # score1_2, score2_2, score3_2, score4_2, score5_2 = sk(x_train2, y_train2, x_test2, y_test2)
    # score1_3, score2_3, score3_3, score4_3, score5_3 = sk(x_train3, y_train3, x_test3, y_test3)
    # if len(x_train4) > 0 and len(x_test4) > 0 and len(y_train4) > 0 and len(y_test4) > 0:
    #     score1_4, score2_4, score3_4, score4_4, score5_4 = sk(x_train4, y_train4, x_test4, y_test4)







    score1, score2, score3, score4, score5, title = [], [], [], [], [], []
    if score1_1 >= 0 or score2_1 >= 0 or score3_1 >= 0 or score4_1 >= 0 or score5_1 >= 0:
        title.append('hungarian')
        score1.append(score1_1)
        score2.append(score2_1)
        score3.append(score3_1)
        score4.append(score4_1)
        score5.append(score5_1)
    if score1_2 >= 0 or score2_2 >= 0 or score3_2 >= 0 or score4_2 >= 0 or score5_2 >= 0:
        title.append('long-beach-va')
        score1.append(score1_2)
        score2.append(score2_2)
        score3.append(score3_2)
        score4.append(score4_2)
        score5.append(score5_2)
    if score1_3 >= 0 or score2_3 >= 0 or score3_3 >= 0 or score4_3 >= 0 or score5_3 >= 0:
        title.append('switzerland')
        score1.append(score1_3)
        score2.append(score2_3)
        score3.append(score3_3)
        score4.append(score4_3)
        score5.append(score5_3)
    if score1_4 >= 0 or score2_4 >= 0 or score3_4 >= 0 or score4_4 >= 0 or score5_4 >= 0:
        title.append('all')
        score1.append(score1_4)
        score2.append(score2_4)
        score3.append(score3_4)
        score4.append(score4_4)
        score5.append(score5_4)

    if len(score1) > 0:
        score = {}
        if score1.count(-1) == 0:
            score.update(KNeighbors=score1)
        if score2.count(-1) == 0:
            score.update(DecisionTree=score2)
        if score3.count(-1) == 0:
            score.update(RandomForest=score3)
        if score4.count(-1) == 0:
            score.update(SVC=score4)
        if score5.count(-1) == 0:
            score.update(Voting=score5)
        frame = pd.DataFrame(data=score, index=title)
        print(frame)

    plt.show()

# 学习曲线
# KNeighborsClassifier()  DecisionTreeClassifier()  RandomForestClassifier() SVC()
# ShowLearnLine(KNeighborsClassifier(), 'knn', set1)
# ShowLearnLine(DecisionTreeClassifier(), 'DT', set1)
# ShowLearnLine(RandomForestClassifier(), 'RF', set1)
# ShowLearnLine(SVC(), 'SVC', set1)


# 验证曲线
# KNeighborsClassifier()  DecisionTreeClassifier()  RandomForestClassifier() SVC()
# ShowValidationLine(KNeighborsClassifier(), 'knn', set1, 'n_neighbors', range(1, 50))
# ShowValidationLine(DecisionTreeClassifier(), 'DT', set1, 'max_depth', range(1, 50))
# ShowValidationLine(RandomForestClassifier(), 'RF', set1, 'max_depth', range(1, 50))
# ShowValidationLine(SVC(), 'SVC', set1, 'kernel', range(1, 50))

# 寻找最佳值
# FindBestParameters(set1[:, range(row_count1)], set1[:, row_count1], 'hungarian', n_jobs=-1)
# FindBestParameters(set2[:, range(row_count2)], set2[:, row_count2], 'long-beach-va', n_jobs=-1)
# FindBestParameters(set3[:, range(row_count3)], set3[:, row_count3], 'switzerland', n_jobs=-1)
# FindBestParameters(set4[:, range(row_count4)], set4[:, row_count4], 'all', n_jobs=-1)

# FeaturePipeLine(KNeighborsClassifier(),x_train_out1, y_train_out1, x_test_out1, y_test_out1)

# if __name__=='__main__':
#     FindBestParameters(set5[:, range(columns_count5 - 1)], set5[:, columns_count5 - 1], 'all', n_jobs=-1)



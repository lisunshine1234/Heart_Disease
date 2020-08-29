# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os  #从文件系统中读取图片
import warnings
import numpy as np
from other import skstudy as ms, skstudy as sm, skstudy as dc
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D   #三维
from other.skstudy import svm
# import neurolab as nl
import matplotlib.pyplot as plt
from other.skstudy import accuracy_score
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.seterr(all='ignore')

#迭代,返回每个文件的路径
def search_objects(directory):
    directory = os.path.normpath(directory)  #路径规格化
    if not os.path.isdir(directory):
        raise IOError("The directory '" + directory + "' doesn't exist!")
    objects = {}  #键：标签，值：该标签下的所有图片
    for curdir, subdirs, files in os.walk(directory):
        for csv in (file for file in files if file.endswith('.csv')):
            path = os.path.join(curdir, csv) #路径拼接
            label = path.split(os.path.sep)[-2]
            if label not in objects:
                objects[label] = []
            objects[label].append(path)
    return objects

objects = search_objects(r'C:\Users\tslee\Numpy_20190214\E-tongue\data\data')
#print(objects.keys())

x, y = [], []
for label, filenames in objects.items():
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            for line in f.readlines()[100:]:
                #取141之后的数据！
                data = [float(substr) for substr in line.split(',')]
                x.append(data)
                y.append(label)

print(len(x))
a = len(x)

x = np.array(x)
y = np.array(y, dtype=int)
print(x.shape, y.shape)
'''
z, v = np.zeros(shape=(int(a / 60),8)), np.zeros(shape=(int(a / 60),))
for i in range(int(a/60)):
    z[i] = np.vsplit(x, int(a / 60))[i].mean(axis=0)
    v[i] = int(i%12)
x = z
y = v
#print(x, y)
print(x.shape, y.shape)
'''


#PCA提取主成分
pca_model = dc.PCA(n_components=5)
z = pca_model.fit_transform(x)
#P_component = np.array(pca_model.components_)
#print('P_component =', x)
#print('explained_variance_ratio:', pca_model.explained_variance_ratio_.sum())   #还原率
'''
#MDS提取降维
similarities = euclidean_distances(x)
seed = np.random.RandomState(seed=3)
mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
x = mds.fit(similarities).embedding_
#pos *= np.sqrt((x ** 2).sum()) / np.sqrt((pos ** 2).sum())
'''

train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.2, random_state=7)
#model = se.RandomForestClassifier(max_depth=4, n_estimators=15)
model = svm.SVC(kernel='sigmoid', C=600, gamma=0.01)  #kernel='rbf'

model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print('Accuracy = ', (pred_test_y == test_y).sum()/test_y.size, sep = '\n')

y_ture = [int(i) for i in test_y]
predictions = [int(i) for i in pred_test_y]
acc = accuracy_score(y_ture, predictions)
print('分类正确率:', acc)

print('Classification Report:', sm.classification_report(test_y, pred_test_y), sep = '\n')   #分类报告

cm = sm.confusion_matrix(test_y, pred_test_y)  #混淆矩阵
print('Confusion_matrix', cm, sep='\n')
mp.figure('Confusion Matrix', facecolor='lightgray')
mp.title('Confusion Matrix', fontsize=20)
mp.xlabel('Predicted Class', fontsize=14)
mp.ylabel('True Class', fontsize=14)
mp.tick_params(labelsize=10)
mp.imshow(cm, interpolation='nearest', cmap='brg')


print(x.shape)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(z[:,0], z[:,1], z[:,2], c=y, cmap='brg', s=60)


#C0, C1 = y == 0, y == 1
#mp.scatter(x[C0][:, 0], x[C0][:, 1], c='orangered', s=60)
#mp.scatter(x[C1][:, 0], x[C1][:, 1], c='limegreen', s=60)
#mp.legend()
mp.show()
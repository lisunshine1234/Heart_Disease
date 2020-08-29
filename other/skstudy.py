from sklearn.feature_selection import VarianceThreshold
import numpy as np

X = np.loadtxt(open("E:/resource/project/test/second_first_hungarian.csv", "r"), delimiter=',', skiprows=0)
selector = VarianceThreshold(1)
selector.fit(X)
print('Variances is %s'%selector.variances_)
print('After transform is \n%s'%selector.transform(X))
print('The surport is %s'%selector.get_support(True))#如果为True那么返回的是被选中的特征的下标
print('The surport is %s'%selector.get_support(False))#如果为FALSE那么返回的是布尔类型的列表，反应是否选中这列特征
print('After reverse transform is \n%s'%selector.inverse_transform(selector.transform(X)))
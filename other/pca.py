from numpy import *
import matplotlib.pyplot as plt
import csv


def loadDataSet():
    datArr = loadtxt(open("E:/resource/project/test/second_first_hungarian.csv", "rb"), delimiter=',', skiprows=0)
    # 利用map()函数，将列表中每一行的数据值映射为float型
    # datArr = [map(float, line) for line in stringArr]
    # 将float型数据值的列表转化为矩阵返回
    return mat(datArr)


# pca特征维度压缩函数
# @dataMat 数据集矩阵
# @topNfeat 需要保留的特征维度，即要压缩成的维度数，默认4096
def pca(dataMat, topNfeat=4096):
    # 求数据矩阵每一列的均值
    meanVals = mean(dataMat, axis=0)
    # 数据矩阵每一列特征减去该列的特征均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
    # cov(X,0) = cov(X) 除数是n-1(n为样本个数)
    # cov(X,1) 除数是n
    covMat = cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值及对应的特征向量
    # 均保存在相应的矩阵中
    eigVals, eigVects = linalg.eig(mat(covMat))
    # sort():对特征值矩阵排序(由小到大)
    # argsort():对特征值矩阵进行由小到大排序，返回对应排序后的索引
    eigValInd = argsort(eigVals)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:, eigValInd]
    # 将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # 返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    # return lowDDataMat,reconMat
    return lowDDataMat, reconMat


def main():
    lowDDataMat, reconMat = pca(loadDataSet(), 2)
    with open('E:\\resource\\project\\test\\lowDDataMat.csv', 'w',
              newline='') as file_csv:
        write_csv = csv.writer(file_csv)
        write_csv.writerows(lowDDataMat.tolist())
    with open('E:\\resource\\project\\test\\reconMat.csv', 'w',
              newline='') as file_csv:
        write_csv = csv.writer(file_csv)
        write_csv.writerows(reconMat.tolist())
    plt.scatter(lowDDataMat[:, 0].tolist(), lowDDataMat[:, 1].tolist(), s=10)

    plt.show()


if __name__ == "__main__":
    main()

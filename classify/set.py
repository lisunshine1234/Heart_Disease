from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from classify.DataProcess import *
# from imblearn.combine import SMOTEENN
import scipy.signal as signal

# data_path = 'D:\\resource\\project\\test\\'
# # data_path = '/home/liziyang/set/'
#
# file1 = "train_hungarian_2.csv"
# file2 = "train_long-beach-va_2.csv"
# file3 = "train_switzerland_2.csv"
#
# set1 = np.loadtxt(open(data_path + file1, "rb"), delimiter=',', skiprows=1, comments='#')
# set2 = np.loadtxt(open(data_path + file2, "rb"), delimiter=',', skiprows=1, comments='#')
# set3 = np.loadtxt(open(data_path + file3, "rb"), delimiter=',', skiprows=1, comments='#')
# set4 = np.vstack((set1, set2, set3))
set4 = np.loadtxt(open("D:\\resource\\project\\test\\train_all_3.csv", "rb"), delimiter=',', skiprows=1, comments='#')

# columns_count1 = len(set1[0, :])
# columns_count2 = len(set2[0, :])
# columns_count3 = len(set3[0, :])
columns_count4 = len(set4[0, :])

# rows_count1 = len(set1)
# rows_count2 = len(set2)
# rows_count3 = len(set3)
rows_count4 = len(set4)

# set1 = SignalProcess(set1, range(6))
# set2 = SignalProcess(set2, range(6))
# set3 = SignalProcess(set3, range(6))
# set4 = SignalProcess(set4, range(6))

# set1 = ToOne(set1)
# set2 = ToOne(set2)
# set3 = ToOne(set3)
set4 = ToOne(set4)

# set1 = shuffle(set1)
# set2 = shuffle(set2)
# set3 = shuffle(set3)
set4 = shuffle(set4)

# x_1 = set1[:, range(columns_count1 - 1)]
# y_1 = set1[:, columns_count1 - 1]
# x_2 = set2[:, range(columns_count2 - 1)]
# y_2 = set2[:, columns_count2 - 1]
# x_3 = set3[:, range(columns_count3 - 1)]
# y_3 = set3[:, columns_count3 - 1]
x_4 = set4[:, range(columns_count4 - 1)]
y_4 = set4[:, columns_count4 - 1]

# x_train1, x_test1, y_train1, y_test1 = train_test_split(x_1, y_1, test_size=0.2, random_state=1)
# x_train2, x_test2, y_train2, y_test2 = train_test_split(x_2, y_2, test_size=0.2, random_state=1)
# x_train3, x_test3, y_train3, y_test3 = train_test_split(x_3, y_3, test_size=0.2, random_state=1)
x_train4, x_test4, y_train4, y_test4 = train_test_split(x_4, y_4, test_size=0.2, random_state=1)
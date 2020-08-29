from classify.set import *
import csv
from glob import glob
import os
import numpy as np

data_path = 'E:\\resource\\project\\test\\'
file_list = glob(data_path + "*.data")
file_list2 = glob(data_path + "first_*.csv")


def DataToCsv():
    for f in file_list:
        with open(f, 'r') as file:
            with open(data_path + 'first_' + os.path.basename(f).split('.data')[0] + '.csv', 'w',
                      newline='') as file_csv:
                matrix1 = []
                matrix2 = []
                write_info = file.read().replace('\n', ' ')
                for data in write_info.split(' '):
                    matrix2.append(data)
                    if (data.isalpha()):
                        matrix1.append(matrix2)
                        matrix2 = []
                write_csv = csv.writer(file_csv)
                write_csv.writerows(matrix1)
                file_csv.close()

                print(os.path.basename(f) + '\t->\tfirst_' + os.path.basename(f).split('.data')[
                    0] + '.csv' + '\t is done!')


def findLostRow(file_list):
    totleall = 0
    totlecount = []
    totlepresent = []

    for f in file_list:
        with open(f, 'r') as csvfile:
            reader = csv.reader(csvfile)
            arr = np.array(list(reader))

            all = len(arr)
            totleall += all
            count = []
            present = []

            x = []
            for i in range(len(arr[0, :])):
                x.append(i + 1)

            for i in range(len(arr[0, :])):
                count.append(np.sum(arr[:, i] == '-9'))
                present.append(np.sum(arr[:, i] == '-9') / all * 100)

            totlecount.append(count)

            hist = pygal.Bar()
            hist.x_labels = x
            hist.x_title = "标签编号"
            hist.y_title = os.path.basename(f) + "百分比"
            hist.add(str(all), present)
            hist.render_to_file(data_path + os.path.basename(f).split('.')[0] + '.svg')
            print(os.path.basename(f) + '\t->\t' + os.path.basename(f).split('.csv')[0] + '.svg\t is done!')

    for i in range(len(totlecount[0])):
        totlepresent.append(np.sum(np.array(totlecount)[:, i]) / totleall * 100)

    hist1 = pygal.Bar()
    hist1.x_labels = x
    hist1.x_title = "标签编号"
    hist1.y_title = "总体百分比"
    hist1.add(str(totleall), totlepresent)
    hist1.render_to_file(data_path + 'all.svg')
    print('all.svg \t is done!')


def RowSelection():
    rows = [10, 37, 34, 35, 32, 33, 31, 16, 12, 19, 30, 41, 40, 5, 6, 7, 9, 38, 13, 14, 15, 17, 18, 3, 4, 58]
    # rows = [10, 37, 32, 33, 31, 16, 12, 19, 30, 41, 40, 58]
    for i in range(len(rows)):
        rows[i] = rows[i] - 1
    for f in file_list2:
        with open(f, 'r') as csvfile:
            reader = csv.reader(csvfile)
            arr = np.array(list(reader))
            with open(data_path + 'second_' + os.path.basename(f).split('.csv')[0] + '.csv', 'w',
                      newline='') as file_csv:
                write_csv = csv.writer(file_csv)
                write_csv.writerows(arr[:, rows])
                print(os.path.basename(f) + '\t->\tsecond_' + os.path.basename(f).split('.csv')[
                    0] + '.csv' + '\t is done!')


def CombineFile():
    # file1 = np.loadtxt(open(data_path + "second_first_hungarian.csv", "rb"), delimiter=',', skiprows=0)
    # file2 = np.loadtxt(open(data_path + "second_first_long-beach-va.csv", "rb"), delimiter=',', skiprows=0)
    # file3 = np.loadtxt(open(data_path + "second_first_switzerland.csv", "rb"), delimiter=',', skiprows=0)

    data_set = np.vstack((set1,set2,set3))

    with open(data_path + 'train.csv', 'w', newline='') as file_csv:
        write_csv = csv.writer(file_csv)
        write_csv.writerows(data_set)
        file_csv.close()

CombineFile()
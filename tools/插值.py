import os

import pandas as pd

for i in range(0, 70):

    csv_path = "/root/hy-tmp/data/data/" + str(i).rjust(3, '0')

    for dirlist in os.listdir(csv_path):
        csv_path_final = csv_path + "/" + dirlist
        if not os.path.exists(csv_path_final):
            continue
        else:
            data = pd.read_csv(csv_path_final, header=0, sep=',')
            # print(data.isnull().sum().sum())
            if data.shape[1] == 93:
                divide_line = 1 / 2 * (data.iloc[:, 1:].shape[0] - 1) * data.iloc[:, 1:].shape[1]
                if data.isnull().sum().sum() <= divide_line and not (data.shape[0] - data.isna().sum() <= 3).any():
                    _data_temp_1 = data.iloc[:, 1:]
                    for j in range(_data_temp_1.shape[1]+1):
                        data.iloc[:, j].interpolate(method='spline', order=3, limit_direction='both', inplace=True)
                    data.to_csv(csv_path_final)
                    print('success process', csv_path_final)
                else:
                    os.remove(csv_path_final)
                    print('delete ', csv_path_final)

            if data.shape[1] == 94:
                divide_line = 1 / 2 * (data.iloc[:, 2:].shape[0] - 1) * data.iloc[:, 2:].shape[1]
                if data.isnull().sum().sum() <= divide_line and not (data.shape[0] - data.isna().sum() <= 3).any():
                    _data_temp_2 = data.iloc[:, 2:]
                    for j in range(_data_temp_2.shape[1]+1):
                        data.iloc[:, j].interpolate(method='spline', order=3, limit_direction='both', inplace=True)
                    data.to_csv(csv_path_final)
                    print('success process', csv_path_final)
                else:
                    os.remove(csv_path_final)
                    print('delete ', csv_path_final)

            # else:
            #     os.remove(csv_path_final)
            #     print('delete ', csv_path_final)

    # Gold.to_csv('1.csv')

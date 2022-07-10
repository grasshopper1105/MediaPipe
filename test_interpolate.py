import os
import pandas as pd
import numpy as np

# csv_path = "D:\\Software\\PyCharm\\AI\\data\\051\\P01_03_11_0.csv"
csv_path = "D:\\Software\\PyCharm\\AI\\data\\000\\P01_01_00_0.csv"

data = pd.read_csv(csv_path, header=0, sep=',')
print((data.shape[0] - data.isna().sum() <= 3).any())
# print(data.isnull().sum().sum())
# for i in range(data.shape[1]):
#     data.iloc[2:, i].interpolate(method='spline', order=2, limit_direction='both', inplace=True)
#     data.to_csv('2.csv')

import os
import pandas as pd

csv_path = "/root/hy-tmp/data/data/000/004.csv"
data = pd.read_csv(csv_path, header=0, sep=',')
# data.iloc[:, -1].fillna(method='bfill', inplace=True)
data.fillna(method='pad', axis=0, inplace=True)
data.to_csv('/root/hy-tmp/1.csv')

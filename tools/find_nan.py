import os
import pandas as pd

for i in range(0, 70):

    csv_path = "D:/Software/PyCharm/AI/data/" + str(i).rjust(3, '0')
    # csv_path = "/root/hy-tmp/data/data/" + str(i).rjust(3, '0')

    for dirlist in os.listdir(csv_path):
        csv_path_final = csv_path + "/" + dirlist
        # print(csv_path_final)
        if not os.path.exists(csv_path_final):
            continue
        else:
            data = pd.read_csv(csv_path_final, header=0, sep=',')
            if data.iloc[1:, :].isnull().any().any():
                # data.fillna(method='pad', inplace=True)
                data.fillna(method='pad', axis=0, inplace=True)
                data.to_csv(csv_path_final)
                print(csv_path_final)

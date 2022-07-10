import os

for j in range(0, 70):
    # path = '/root/hy-tmp/data/data/' + str(j).rjust(3, '0')
    path = "D:/Software/PyCharm/AI/data/" + str(j).rjust(3, '0')
    files = os.listdir(path)  # 文件夹里的所有文件名存成列表list
    for i, file in enumerate(files):
        # 重点在05d，这样会自动补齐5位，不足的补零
        # 为啥是0 + i，方便后面添加，把0改了就行
        NewFileName = os.path.join(path, '%03d' % (0 + i) + '.csv')
        OldFileName = os.path.join(path, file)
        print('第%d个文件：%s' % (i + 1, NewFileName))
        os.rename(OldFileName, NewFileName)  # 改名

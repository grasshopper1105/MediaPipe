import zipfile
import os
import shutil


def unzip_file(path):
    filenames = os.listdir(path)  # 获取目录下所有文件名
    for filename in filenames:
        filepath = os.path.join(path, filename)
        zip_file = zipfile.ZipFile(filepath)  # 获取压缩文件
        # print(filename)
        new_filepath = filename.split(".", 1)[0]  # 获取压缩文件的文件名
        new_filepath = os.path.join(path, new_filepath)
        # print(new-filepath)
        if os.path.isdir(new_filepath):  # 根据获取的压缩文件的文件名建立相应的文件夹
            pass
        else:
            os.mkdir(new_filepath)
        for name in zip_file.namelist():  # 解压文件
            zip_file.extract(name, new_filepath)
        zip_file.close()
        Conf = os.path.join(new_filepath, 'conf')
        if os.path.exists(Conf):  # 如存在配置文件，则删除（需要删则删，不要的话不删）
            shutil.rmtree(Conf)
        if os.path.exists(filepath):  # 删除原先压缩包
            os.remove(filepath)
        print("解压{0}成功".format(filename))


if __name__ == '__main__':
    unzip_file('/hy-tmp/Insect/data')

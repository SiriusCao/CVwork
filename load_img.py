from torch.utils.data import Dataset
import os
import cv2
import numpy as np


class MyDataset(Dataset):
    def __init__(self, train_path, transform=None):
        self.images = os.listdir(train_path + '/last')
        self.labels = os.listdir(train_path + '/last_msk')
        # 如果images好labels的数量不一样，则抛出异常
        assert len(self.images) == len(self.labels), 'Number does not match'
        self.transform = transform
        self.images_and_labels = []    # 存储图像和标签路径
        for i in range(len(self.images)):
            self.images_and_labels.append((train_path + '/last/' + self.images[i], train_path + '/last_msk/' + self.labels[i]))

    def __getitem__(self, item):
        img_path, lab_path = self.images_and_labels[item]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        lab = cv2.imread(lab_path, 0)   #第二个参数是一个标识，它用来指定图像的读取方式，0表示加载一张灰度图
        lab = cv2.resize(lab, (224, 224))
        lab = lab / 255    # 转换成0和1
        lab = lab.astype('uint8')    # 不为1的全置为0
        # eye(N,M=None, k=0, dtype=<type ‘float’>)
        # 一个参数：输出方阵（行数=列数）的规模，即行数或列数
        # 第三个参数：默认情况下输出的是对角线全“1”，其余全“0”的方阵
        lab = np.eye(2)[lab]    # one-hot编码
        # 根据提供的lambda表达式对lab做映射
        # 将所有0变为1（1对应255， 白色背景），所有1变为0（黑色，目标）
        lab = np.array(list(map(lambda x: abs(x-1), lab))).astype('float32')
        lab = lab.transpose(2, 0, 1)  # [224, 224, 2] => [2, 224, 224]
        if self.transform is not None:
            img = self.transform(img)
        return img, lab

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    img = cv2.imread('data/train/last_msk/150.jpg', 0)
    img = cv2.resize(img, (16, 16))
    img2 = img/255
    img3 = img2.astype('uint8')
    hot1 = np.eye(2)[img3]
    hot2 = np.array(list(map(lambda x: abs(x-1), hot1)))
    print(hot2.shape)
    print(hot2.transpose(2, 0, 1))

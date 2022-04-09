import os
import model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from load_img import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader


batchsize = 8
epochs = 50
train_data_path = 'data/train'

# 常用的图像预处理方法
transform = transforms.Compose([transforms.ToTensor(),  # 图片转张量
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# 构建MyDataset实例
bag = MyDataset(train_data_path, transform)
# 构建DataLoder
dataloader = DataLoader(bag, batch_size=batchsize, shuffle=True)


device = torch.device('cuda')
net = model.Net().to(device)
# BinaryCrossEntropyLoss 对一个batch里面的数据做二元交叉熵
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.7)

# 如果不存在该目录则创建
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

for epoch in range(1, epochs+1):
    for batch_idx, (img, lab) in enumerate(dataloader):
        img, lab = img.to(device), lab.to(device)
        output = torch.sigmoid(net(img))
        loss = criterion(output, lab)

        output_np = output.cpu().data.numpy().copy()
        output_np = np.argmin(output_np, axis=1)
        y_np = lab.cpu().data.numpy().copy()
        y_np = np.argmin(y_np, axis=1)

        # 每20个batch打印一次日志
        if batch_idx % 20 == 0:
            print('Epoch:[{}/{}]\tStep:[{}/{}]\tLoss:{:.6f}'.format(
                epoch, epochs, (batch_idx+1)*len(img), len(dataloader.dataset), loss.item()
            ))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每十次epoch保存一次训练好的模型
    if epoch % 10 == 0:
        torch.save(net, 'checkpoints/model_epoch_{}.pth'.format(epoch))
        print('checkpoints/model_epoch_{}.pth saved!'.format(epoch))
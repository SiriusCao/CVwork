import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # torch.nn.Sequential是一个容器，模块将按照构造函数中传递的顺序添加到模块中
        self.encode1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 输入的通道数目、输出的通道数目、卷积核的大小、卷积每次滑动的步长为多少、padding
            nn.BatchNorm2d(64),  # 使一批（Batch）feature map满足均值为0，方差为1的分布规律。这样不仅数据分布一致，而且避免发生梯度消失
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # max pooling的窗口大小、max pooling的窗口移动的步长
        )
        self.encode2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.encode3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.encode4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.encode5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.decode1 = nn.Sequential(
            # nn.ConvTranspose2d的功能是进行反卷积操作
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decode4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decode5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.classifier = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):           # b: batch_size
        out = self.encode1(x)       # [b, 3, 224, 224]  =>  [b, 64, 112, 112]
        out = self.encode2(out)     # [b, 64, 112, 112] =>  [b, 128, 56, 56]
        out = self.encode3(out)     # [b, 128, 56, 56]  =>  [b, 256, 28, 28]
        out = self.encode4(out)     # [b, 256, 28, 28]  =>  [b, 512, 14, 14]
        out = self.encode5(out)     # [b, 512, 14, 14]  =>  [b, 512, 7, 7]
        out = self.decode1(out)     # [b, 512, 7, 7]    =>  [b, 256, 14, 14]
        out = self.decode2(out)     # [b, 256, 14, 14]  =>  [b, 128, 28, 28]
        out = self.decode3(out)     # [b, 128, 28, 28]  =>  [b, 64, 56, 56]
        out = self.decode4(out)     # [b, 64, 56, 56]   =>  [b, 32, 112, 112]
        out = self.decode5(out)     # [b, 32, 112, 112] =>  [b, 16, 224, 224]
        out = self.classifier(out)  # [b, 16, 224, 224] =>  [b, 2, 224, 224]   2表示类别数，目标和非目标两类
        return out


if __name__ == '__main__':
    img = torch.randn(2, 3, 224, 224)
    net = Net()
    sample = net(img)
    print(sample.shape)
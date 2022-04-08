import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os


class TestDataset(Dataset):
    def __init__(self, test_img_path, transform=None):
        # self.test_img = os.listdir(test_img_path)
        self.name=test_img_path
        self.transform = transform
        self.images = []
        # for i in range(len(self.test_img)):
        self.images.append(test_img_path)

    def pic_name(self):
        return self.name

    def __getitem__(self, item):
        img_path = self.images[item]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)


def main(test_img_path):
    # test_img_path = 'data/test/last/502.jpg'
    checkpoint_path = 'checkpoints/model_epoch_50.pth'

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    bag = TestDataset(test_img_path, transform)
    name=bag.pic_name()
    dataloader = DataLoader(bag, batch_size=1, shuffle=None)

    net = torch.load(checkpoint_path)
    net = net.cuda()
    for idx, img in enumerate(dataloader):
        img = img.cuda()
        output = torch.sigmoid(net(img))

        output_np = output.cpu().data.numpy().copy()
        output_np = np.argmin(output_np, axis=1)

        img_arr = np.squeeze(output_np)
        img_arr = img_arr*255

        true_name=str(name).split('/')[-1]
        cv2.imwrite(true_name, img_arr)
        print(true_name)

if __name__ == '__main__':
    main('data/test/last/501.jpg')
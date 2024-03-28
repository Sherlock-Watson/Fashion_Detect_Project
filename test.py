import numpy as np
import torch
import fashion_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
train_attr_file = "dataset/split/train_attr.txt"
train_attr = np.loadtxt(train_attr_file, dtype=int)
train_attr = torch.from_numpy(train_attr)
num_classes = torch.max(train_attr, dim=0).values + 1
print(num_classes)

# model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
#
# # 打印模型结构
# print(model)
#
# # 查看模型的最后一层
# print(model.classifier)
#
# # 获取最后一层全连接层的输入特征数量
# num_features = model.classifier.in_features
# print("最后一层全连接层的输入特征数量:", num_features)

# path = "dataset/split/train.txt"
# root = "dataset"
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((128, 128))
# ])
#
# dataset = fashion_dataset.FashionNet_Dataset(root, path, transform)
#
# train_loader = DataLoader(dataset, 16, shuffle=True)
#
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
#
# rand_img = np.transpose(images[0].cpu().detach().numpy(), (1, 2, 0))
# plt.imshow(rand_img)
# plt.show()
# print(f"dataset size: {len(dataset)}")
#
# print(dataset[0][1])
# mean = np.array([0.0, 0.0, 0.0])
# img = np.array(dataset[0][0])
# img_h = img.shape[1]
# img_w = img.shape[2]
# standardized_pixel_count = img_w * img_h
#
# print(f"imgh: {img_h}; imgw: {img_w}")
# for train_item, target in dataset:
#     train_item = np.array(train_item)
#     mean += np.sum(train_item, axis=(1, 2))
#
# mean = [m / len(dataset) for m in mean]


# # get mean and deviation
# print(f"mean: {mean}")
# print(f"standard deviation: {std}")
#
# print('Mean of each color channel:', mean)
# print('Standard deviation of each color channel:', std)
#
# # get the new transform
# train_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomCrop(size=32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)
# ])
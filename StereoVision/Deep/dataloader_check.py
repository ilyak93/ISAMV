import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from dataloaders import CustomImageDataset

dataset = CustomImageDataset(
    img_dir="E:/dataset/left_right_dis_dep/"
)
lengths = [3198, len(dataset)-3198]
train_set, val_set = torch.utils.data.random_split(dataset, lengths)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(val_set, batch_size=64, shuffle=False)

[left_samples, right_samples], [dis, depths] = next(iter(train_dataloader))

left_sample = left_samples[5]
right_sample = right_samples[5]
dis = dis[5]
depth = depths[5]
plt.imshow(left_sample)
plt.show()
plt.imshow(right_sample)
plt.show()
plt.imshow(dis)
plt.show()
plt.imshow(depth)
plt.show()
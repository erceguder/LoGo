# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image, ImageFilter
import random

# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
def augmentation(resolution, scale):
    return transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

class CIFAR10Pair(CIFAR10):
    def __init__(self, root, train, download, r_g=0.65, r_l=0.4):
        super().__init__(root, train, transform=None, download=download)

        self.global_transform = augmentation(32, (r_g, 0.9))
        self.local_transform = augmentation(32, (0.15, r_l))

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        g1 = self.global_transform(img)
        g2 = self.global_transform(img)
        l1 = self.local_transform(img)
        l2 = self.local_transform(img)

        return g1, g2, l1, l2

# class TwoCropsTransform:
#     """Take two random crops of one image as the query and key."""

#     def __init__(self, base_transform):
#         self.base_transform = base_transform

#     def __call__(self, x):
#         q = self.base_transform(x)
#         k = self.base_transform(x)
#         return [q, k]


# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma

#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x

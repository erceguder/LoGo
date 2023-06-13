from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

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
    def __init__(self, root, train, download, r_g=0.6, r_l=0.2):
        super().__init__(root, train, transform=None, download=download)

        self.global_transform = augmentation(32, (r_g, 0.9))
        self.local_transform = augmentation(32, (0.05, r_l))

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        global_crops = self.global_transform(img), self.global_transform(img)
        local_crops = self.local_transform(img), self.local_transform(img)

        return global_crops, local_crops
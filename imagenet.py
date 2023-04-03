#File using for ImageNet

import os
from torch.utils.data import (Dataset,
                              DataLoader,
                              TensorDataset)
from torchvision.datasets import ImageFolder
from torchvision import transforms


traindir = os.path.join('/home/dnn/nas/ImageNet1', 'ILSVRC2012_img_train')
valdir = os.path.join('/home/dnn/nas/ImageNet1', 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

val_dataset = ImageFolder(valdir,
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True,
            num_workers=16, pin_memory=True)

val_loader = DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=16, pin_memory=True)

import os
from tkinter.tix import Tree
import torchvision

from torchvision.datasets.utils import download_and_extract_archive


def create_dataset(name, is_training):
    torchvision.datasets.MNIST('./data/MNIST', train=True, download=True) # workaround to create the correct folder directories
    name = name.lower()

    if name == 'mnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(3)
        ])
        ds = torchvision.datasets.MNIST('./data/mnist', train=is_training, download=True, transform=transforms)
    elif name == 'usps':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(3)
        ])
        ds = torchvision.datasets.USPS('./data/usps', train=is_training, download=True, transform=transforms)
    elif name == 'visda':
        if not (os.path.isdir('./data/visda/train') or os.path.isdir('./data/visda/validation') or os.path.isdir('./data/visda/test')):
            if not os.path.isdir('./data/visda'):
                os.mkdir('./data/visda')
            if not os.path.isdir('./data/visda/train'):
                download_and_extract_archive(url='http://csr.bu.edu/ftp/visda17/clf/train.tar',
                         download_root='./data/visda',
                         extract_root='./data/visda',
                         filename='train.tar'
                )
            if not os.path.isdir('./data/visda/validation'):
                download_and_extract_archive(url='http://csr.bu.edu/ftp/visda17/clf/validation.tar',
                         download_root='./data/visda',
                         extract_root='./data/visda',
                         filename='validation.tar'
                )
        if is_training:
            transforms = torchvision.transforms.Compose([
              torchvision.transforms.ToTensor()
            ])
            
            ds = torchvision.datasets.ImageFolder('./data/visda/train')
        else:
            transforms = torchvision.transforms.Compose([
              torchvision.transforms.ToTensor()
            ])
            ds = torchvision.datasets.ImageFolder('./data/visda/validation')
    elif name == 'cifar10':
        ds = torchvision.datasets.CIFAR10('./data/cifar10', train=is_training, download=True)
    elif name == 'cifar100':
        ds = torchvision.datasets.CIFAR100('./data/cifar100', train=is_training, download=True)
    return ds

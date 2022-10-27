import os
from tkinter.tix import Tree
import torchvision

from torchvision.datasets.utils import download_and_extract_archive

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCExperience, NCScenario



def create_dataset(name, is_training, args, force_unique_tasks=False):
    torchvision.datasets.MNIST('./data/MNIST', train=True, download=True) # workaround to create the correct folder directories
    name = name.lower()
    transforms = None
    tasks = 1 if force_unique_tasks else args.tasks

    if name == 'mnist':
        ds = torchvision.datasets.MNIST('./data/mnist', train=is_training, download=True, transform=transforms)
        ds = nc_benchmark(ds, ds, n_experiences=tasks, seed=args.seed, task_labels=True,
                          fixed_class_order=range(0, 10), train_transform=transforms, eval_transform=transforms)
    elif name == 'usps':
        ds = torchvision.datasets.USPS('./data/usps', train=is_training, download=True)
        ds = nc_benchmark(ds, ds, n_experiences=tasks, seed=args.seed, task_labels=True,
                          fixed_class_order=range(0, 10), train_transform=transforms, eval_transform=transforms)
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
            ds = torchvision.datasets.ImageFolder('./data/visda/train')
            ds = nc_benchmark(ds, ds, n_experiences=tasks, seed=args.seed, task_labels=True,
                              fixed_class_order=range(0, 12), train_transform=transforms, eval_transform=transforms)
        else:
            ds = torchvision.datasets.ImageFolder('./data/visda/validation')
            ds = nc_benchmark(ds, ds, n_experiences=tasks, seed=args.seed, task_labels=True,
                              fixed_class_order=range(0, 12), train_transform=transforms, eval_transform=transforms)
    elif name == 'cifar10':
        ds = torchvision.datasets.CIFAR10('./data/cifar10', train=is_training, download=True)
    elif name == 'cifar100':
        ds = torchvision.datasets.CIFAR100('./data/cifar100', train=is_training, download=True)
    return ds

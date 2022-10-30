import torch.utils.data as data

from PIL import Image
import os
import os.path
import random

def default_loader(path):
	return Image.open(path).convert('RGB')

def default_flist_reader(flist):
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			impath, imlabel = line.strip().split()
			imlist.append( (impath, int(imlabel)) )
					
	return imlist

class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)		
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
		self.targets = [s[1] for s in self.imlist]
		#workaround because painting train is missing one class
		while len(set(self.targets)) < 345:
			missing_target = max([(i if i not in set(self.targets) else 0) for i in range(0,345)])
			random_pos = random.randint(0, len(self.targets))
			self.targets[random_pos] = missing_target # This is a mislabel, but it is also a way to automatically solve this problem for now

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(f'./data/domainnet/{impath}')
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return img, target

	def __len__(self):
		return len(self.imlist)
import torch
import os
from PIL import Image
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    if isinstance(dir,list):
        for i in range(len(dir)):
            dir_i = dir[i]
            for root, _, fnames in sorted(os.walk(dir_i, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
    else:
        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images[:min(max_dataset_size, len(images))]

class basedataset(torch.utils.data.Dataset):
    def __init__(self,path,transform):
        self.paths = sorted(make_dataset(path))
        self.size = len(self.paths)
        self.transform = transform
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img

class namedataset(torch.utils.data.Dataset):
    def __init__(self,path,transform):
        #align multi-domain
        if isinstance(path, list):
            self.paths = []
            for domain_i in range(len(path)):
                path_i = sorted(make_dataset(path[domain_i]))
                self.paths = self.paths + path_i
        else:
            self.paths = sorted(make_dataset(path))
        self.size = len(self.paths)
        self.transform = transform
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.paths[index]
        name = path.split('/')[-1]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img,name

class labeldataset(torch.utils.data.Dataset):
    def __init__(self,paths,transform):
        self.paths = []
        label = []
        for domain_i in range(len(paths)):
            path_i = sorted(make_dataset(paths[domain_i]))
            self.paths = self.paths + path_i
            label_i = domain_i * torch.ones(len(path_i))
            label.append(label_i)
        self.label = torch.cat(label)
        self.size = len(self.paths)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img,label

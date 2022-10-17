import torch
import torchvision.transforms as transforms
from .basedataset import namedataset,labeldataset
from PIL import Image

def get_dataset(phase,image_size,data_path):
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size,interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.Resize(image_size,interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    if phase == 'train':
        dataset = labeldataset(
            data_path,
            transform = train_transform,
        )
    else:
        dataset = namedataset(
            data_path,
            transform=test_transform,
        )
    return dataset

def rescale(X):
    X = 2 * X - 1.0
    return X

def inverse_rescale(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)

def imageresize2tensor(path,image_size):
    img = Image.open(path)
    convert = transforms.Compose(
        [transforms.Resize(image_size,interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    return convert(img)

def image2tensor(path):
    img = Image.open(path)
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img)

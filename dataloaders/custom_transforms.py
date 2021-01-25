import random

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        mask = sample['label']
        mask = np.array(mask).astype(np.float32)

        sample['image'] = img
        sample['label'] = mask
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        mask = sample['label']
        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).float()

        sample['image'] = img
        sample['label'] = mask
        return sample

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = sample['label']
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        sample['image'] = img
        sample['label'] = mask
        return sample

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        sample['image'] = img
        return sample

class RandomScaleCrop(object):
    def __init__(self, baseSize, cropSize, fill=0):
        self.baseSize = baseSize
        self.cropSize = cropSize
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.baseSize * 0.5), int(self.baseSize * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.cropSize:
            padh = self.cropSize - oh if oh < self.cropSize else 0
            padw = self.cropSize - ow if ow < self.cropSize else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.cropSize)
        y1 = random.randint(0, h - self.cropSize)
        img = img.crop((x1, y1, x1 + self.cropSize, y1 + self.cropSize))
        mask = mask.crop((x1, y1, x1 + self.cropSize, y1 + self.cropSize))

        sample['img'] = img
        sample['label'] = mask
        return sample

class FixScaleCrop(object):
    def __init__(self, cropSize):
        self.cropSize = cropSize

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.cropSize
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.cropSize
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.cropSize) / 2.))
        y1 = int(round((h - self.cropSize) / 2.))
        img = img.crop((x1, y1, x1 + self.cropSize, y1 + self.cropSize))
        mask = mask.crop((x1, y1, x1 + self.cropSize, y1 + self.cropSize))

        sample['image'] = img
        sample['label'] = mask
        return sample

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        img = img.resize(self.size, Image.BILINEAR)
        mask = sample['label']
        mask = mask.resize(self.size, Image.NEAREST)

        sample['image'] = img
        sample['label'] = mask
        return sample

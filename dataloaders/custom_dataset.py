# custom_dataset.py
# Aditi Iyer, 01/12/2021
# iyera@mskcc.org
# Description: Custom dataloader
#
#Copyright 2010, Joseph O. Deasy, on behalf of the CERR development team.
#This file is part of The Computational Environment for Radiotherapy Research (CERR).
#CERR development has been led by:  Aditya Apte, Divya Khullar, James Alaly, and Joseph O. Deasy.
#CERR has been financially supported by the US National Institutes of Health under multiple grants.
#CERR is distributed under the terms of the Lesser GNU Public License.
#
#This version of CERR is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#CERR is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License
#along with CERR.  If not, see <http://www.gnu.org/licenses/>.


import os
import numpy as np
from PIL import Image
import h5py
from torch.utils import data
from torchvision import transforms
from skimage.transform import resize
from dataloaders import custom_transforms as tr

class customData(data.Dataset):
# Custom dataset class

    def __init__(self,args,split):

        # Get path to H5 dataset
        self.root = args["inputH5Path"]
        self.images_base = os.path.join(self.root, split)
        self.annotations_base = os.path.join(self.images_base, 'Masks/')
        self.transform = args["imageTransform"]

        # Get file list
        self.files = {}
        self.split = split

        self.files[split] = self.glob(rootdir=self.images_base, suffix='.h5')

        print("Found %d images in dir %s" % (len(self.files[split]),split))

    def __len__(self):
        """ Get no. images in split
        """
        return len(self.files[self.split])

    def __getitem__(self, index):
        """ Get normalized scan & associated label mask
        """
        img, target, fname = self._get_processed_img(index)
        sample = {'image': img, 'label': target, 'fname': fname}

        #Mean & std dev. normalization
        split = self.split.lower()
        if split == "train":
           return self.transform_tr(sample)
        elif split == "val":
           return self.transform_val(sample)
        elif split == "test":
           return self.transform_ts(sample)

    def _get_processed_img(self, index):
        """ Load scan & normalize intensities to range (0-255)
        """
        img_path = self.files[self.split][index].rstrip()
        dirname, fname = os.path.split(img_path)
        img = self._load_image(img_path)


        mask_fname = fname.replace("scan_","")
        mask_fname = mask_fname.replace("slice_","slice")
        lbl_path = os.path.join(self.annotations_base, mask_fname)
        target = self._load_mask(lbl_path)

        return img, target, fname

    def _load_image(self, img_path):
        """Load specified image and return a [H,W,C] Numpy array.
        """
        # Read H5 image
        hf = h5py.File(img_path, 'r')
        im = hf['/scan'][:]

        #Normalize image from 0-255 (to match pre-trained dataset of RGB images with intensity range 0-255)
        image = np.array(im)
        image = image.reshape(im.shape).transpose()

        image = (255*(image - np.min(image)) / np.ptp(image).astype(int)).astype(np.uint8)
        image = Image.fromarray(image.astype(np.uint8))

        #Transformations from JSON?

        return image

    def _load_mask(self, lbl_path):
        """Load label mask
        """

        hf = h5py.File(lbl_path, 'r')
        m1 = hf['/mask'][:]
        m = np.array(m1)
        m = m.reshape(m1.shape).transpose()
        m = Image.fromarray(m.astype(np.uint8))
        # Transformations from JSON?
        return m

    def glob(self, rootdir='.', suffix=''):
        """Performs glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(rootdir, filename)
            for filename in os.listdir(rootdir) if filename.endswith(suffix)]

    def transform_tr(self, sample):
        """Image transformations for training"""
        targs = self.transform
        method = targs["method"]
        pars = targs["parameters"]
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=pars["outSize"]),
            tr.RandomRotate(degree=(90)),
            tr.RandomScaleCrop(baseSize=pars["baseSize"], cropSize=pars["outSize"], fill=255),
            tr.Normalize(mean=pars["mean"], std=pars["std"]),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        """Image transformations for validation"""
        targs = self.transform
        method = targs["method"]
        pars = targs["parameters"]
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=pars["outSize"]),
            tr.FixScaleCrop(cropSize=pars["outSize"]),
            tr.Normalize(mean=pars["mean"], std=(pars["std"])),
            tr.ToTensor()])

        return composed_transforms(sample)


    def transform_ts(self, sample):
        """Image transformations for testing"""
        targs = self.transform
        method = targs["method"]
        pars = targs["parameters"]
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=pars["outSize"]),
            tr.Normalize(mean=pars["mean"], std=pars["std"]),
            tr.ToTensor()])

        return composed_transforms(sample)

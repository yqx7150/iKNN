from __future__ import print_function, division
import os, random, time
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from glob import glob
from PIL import Image as PILImage
import numbers
# from scipy.misc import imread
from scipy.io import loadmat
import os

IMG_EXTENSIONS = ['.jpg', 'JPG', '.mat']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(path):
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    return sorted(images)


def loadmri(path):
    mri_images = loadmat(path)['Img']
    img = np.real(mri_images)
    img = np.array(img, dtype=np.float32)

    return img


def loadmri_real_imag(path):
    # not cross real and imag
    mri_images_temp = loadmat(path)['Img']
    mri_images = np.zeros(mri_images_temp.shape, dtype=np.complex128)
    for i in range(mri_images_temp.shape[-1]):
        mri_images[:, :, i] = np.fft.fftshift(np.fft.fft2(mri_images_temp[:, :, i]))
    img_real = np.real(mri_images)
    img_real = np.array(img_real, dtype=np.float32)
    img_imag = np.imag(mri_images)
    img_imag = np.array(img_imag, dtype=np.float32)
    img = np.zeros([256, 256, 2 * (img_real.shape[-1])], dtype=np.float32)
    img = np.concatenate([img_real, img_imag], axis=2)

    return img

def loadmri_T12_cross(path1,path2,path_mask):
    # load data
    mri_images_temp_T1 = loadmat(path1)['Img']
    mri_images_temp_T2 = loadmat(path2)['Img']
    mri_images_temp_mask = loadmat(path_mask)['mask']

    #fft
    mri_images_temp = mri_images_temp_T1.shape
    mri_k_temp_T12 = np.zeros((mri_images_temp[0],mri_images_temp[1],2), dtype=np.complex128)
    mri_k_temp_T12[:, :,0] = np.fft.fftshift(np.fft.fft2(mri_images_temp_T1[:, :]))
    mri_k_temp_T12[:, :, 1] = np.fft.fftshift(np.fft.fft2(mri_images_temp_T2[:, :]))

    #T2 kdata * mask
    mri_k_temp_T12_s = np.zeros((mri_images_temp[0], mri_images_temp[1], 2), dtype=np.complex128)
    mri_k_temp_T12_s[:,:,0] = mri_k_temp_T12[:,:,0]
    mri_k_temp_T12_s[:, :, 1] = np.multiply(mri_images_temp_mask[:,:],mri_k_temp_T12[:,:,1])

    #real imag
    img_real = np.real(mri_k_temp_T12_s)
    img_real = np.array(img_real, dtype=np.float32)
    img_imag = np.imag(mri_k_temp_T12_s)
    img_imag = np.array(img_imag, dtype=np.float32)
    img = np.zeros([256, 256, 2 * (img_real.shape[-1])], dtype=np.float32)
    for i in range(0, 2 * (img_real.shape[-1]), 2):
        img[:, :, i] = img_real[:, :, int(i / 2)]
        img[:, :, i + 1] = img_imag[:, :, int(i / 2)]

    return img

def loadmri_T2_cross(path):
    mri_images_temp_T2 = loadmat(path)['Img']
    mri_k_temp_T2 =  np.fft.fftshift(np.fft.fft2(mri_images_temp_T2[:, :]))
    mri_images_temp = mri_images_temp_T2.shape
    mri_k_temp_T2_2ch = np.zeros((mri_images_temp[0],mri_images_temp[1],2), dtype=np.complex128)
    mri_k_temp_T2_2ch[:,:,0 ] = mri_k_temp_T2
    mri_k_temp_T2_2ch[:, :, 1] = mri_k_temp_T2

    # real imag
    img_real = np.real(mri_k_temp_T2_2ch)
    img_real = np.array(img_real, dtype=np.float32)
    img_imag = np.imag(mri_k_temp_T2_2ch)
    img_imag = np.array(img_imag, dtype=np.float32)
    img = np.zeros([256, 256, 2 * (img_real.shape[-1])], dtype=np.float32)
    for i in range(0, 2 * (img_real.shape[-1]), 2):
        img[:, :, i] = img_real[:, :, int(i / 2)]
        img[:, :, i + 1] = img_imag[:, :, int(i / 2)]


    return img

def loadmri_real_imag_cross(path):
    # cross real and imag
    mri_images_temp = loadmat(path)['Img']
    mri_images = np.zeros(mri_images_temp.shape, dtype=np.complex128)
    for i in range(mri_images_temp.shape[-1]):
        mri_images[:, :, i] = np.fft.fftshift(np.fft.fft2(mri_images_temp[:, :, i]))
    img_real = np.real(mri_images)
    img_real = np.array(img_real, dtype=np.float32)
    img_imag = np.imag(mri_images)
    img_imag = np.array(img_imag, dtype=np.float32)
    img = np.zeros([256, 256, 2 * (img_real.shape[-1])], dtype=np.float32)
    for i in range(0, 2 * (img_real.shape[-1]), 2):
        img[:, :, i] = img_real[:, :, int(i / 2)]
        img[:, :, i + 1] = img_imag[:, :, int(i / 2)]

    return img


class mriDataset_real_imag_cross(Dataset):
    def __init__(self, root1, root2, root):
        self.path_channel_T1_mri = get_image_paths(root1)
        self.path_channel_T2_mri = get_image_paths(root2)
        self.path_channel_mask = root
        self.datanames = np.array([root + "/" + x for x in os.listdir(root1)])

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2, 0, 1)

    def __len__(self):
        return len(self.datanames)

    def __getitem__(self, index):

        # 加载T1_T2_mask
        path_channel_T1 = self.path_channel_T1_mri[index]
        path_channel_T2 = self.path_channel_T2_mri[index]
        path_channel_mask = self.path_channel_mask

        channel_T12_name = path_channel_T1.split('/')[-1].split('.')[0]

        channel4_T12_mri = loadmri_T12_cross(path_channel_T1,path_channel_T2,path_channel_mask)
        channel4_T2_mri = loadmri_T2_cross(path_channel_T2)


        input_channel4_T12_mri = channel4_T12_mri.copy()
        target_channel4_T2_mri = channel4_T2_mri.copy()

        # 将以上图像均转换为tensor形式，变为need的形式
        input_channel4_T12_mri = self.np2tensor(input_channel4_T12_mri)
        target_channel4_T2_mri = self.np2tensor(target_channel4_T2_mri)
        # channel8x3_mri = self.np2tensor(channel8x3_mri)

        sample = {'input_channel4_T12_mri': input_channel4_T12_mri,
                  'target_channel4_T2_mri': target_channel4_T2_mri,
                  'input_channel4_T2s_mri': -1,
                  'channelname': channel_T12_name}

        return sample

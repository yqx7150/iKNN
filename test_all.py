import datetime

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd

import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from model.model import InvISPNet
# from dataset.mri_dataset import mriDataset, mriDataset12, mriDataset12_real_imag, mriDataset12_real_imag_cross
from dataset.mri_dataset import mriDataset12and4_real_imag_cross
from config.config import get_arguments
from utils.commons import denorm, preprocess_test_patch
from tqdm import tqdm
from skimage.measure import compare_psnr, compare_ssim
from scipy.io import savemat
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
channel_pair_list = [[8, 2],
                     [8, 3],
                     [8, 4],
                     [6, 2],
                     [6, 3],
                     ]

torch.set_num_threads(4)
firstTime=1

def WriteInfo_zl(path, **args):
    ppp = os.path.split(path)
    # 如果不存在则创建目录
    if not os.path.isdir(ppp[0]):
        os.makedirs(ppp[0])
        # print(f"{pathDir} 创建成功")

    try:
        args = args['args']
    except:
        pass
    # in any case, don't delete this code ↓
    args['Time'] = [str(datetime.datetime.now())[:-7]]
    try:
        df = pd.read_csv(path, encoding='utf-8', engine='python')
    except:
        df = pd.DataFrame()
    df2 = pd.DataFrame(args)
    df = df.append(df2)
    df.to_csv(path, index=False)

for channel_pair in channel_pair_list:
    channel_ONE = channel_pair[0]
    channel_TWO = channel_pair[1]
    bs = channel_ONE // channel_TWO
    channel_in = int(channel_ONE * 2)
    channel_last = int(channel_TWO * 2)
    modelNum = '0060' # latest


    parser = get_arguments()
    parser.add_argument("--ckpt", type=str, default=f'./exps/{channel_ONE}to{channel_TWO}/checkpoint/{modelNum}.pth',
                        help="Checkpoint path.")
    parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save results. ")
    parser.add_argument("--task", type=str, default=f"test")
    parser.add_argument("--split_to_patch", dest='split_to_patch', action='store_true', help="Test on patch. ")
    args = parser.parse_args()

    ckpt_name = args.ckpt.split("/")[-1].split(".")[0]
    if args.split_to_patch:
        os.makedirs(args.out_path + "%s/results_metric_%s/" % (args.task, ckpt_name), exist_ok=True)
        out_path = args.out_path + "%s/results_metric_%s/" % (args.task, ckpt_name)
    else:
        os.makedirs(args.out_path + "%s/results_%s/" % (args.task, ckpt_name), exist_ok=True)
        out_path = args.out_path + "%s/results_%s/" % (args.task, ckpt_name)


    # def main(args):
    net = InvISPNet(channel_in=channel_in, channel_out=channel_in, block_num=8)
    device = torch.device("cuda:0")
    net.to(device)
    net.eval()
    if os.path.isfile(args.ckpt):
        net.load_state_dict(torch.load(args.ckpt), strict=False)
        print("[INFO] Loaded checkpoint: {}".format(args.ckpt))

    print("[INFO] Start data load and preprocessing")
    # mri_dataset = mriDataset12and4_real_imag_cross(root1='./data/test_data/test_12ch',root2='./data/test_data/test_4ch',root='./data/test_data/test_12ch')
    mri_dataset = mriDataset12and4_real_imag_cross(
        root1=f'/home/liuqieg/code/ZL/MRI/VCA_HG/dataset/test_{channel_ONE}ch_GCC',
        root2='/anyway',
        root=f'/home/liuqieg/code/ZL/MRI/VCA_HG/dataset/test_{channel_ONE}ch_GCC')
    dataloader = DataLoader(mri_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    print("[INFO] Start test...")
    PSNR_COMPRESS = []
    SSIM_COMPRESS = []
    for i_batch, sample_batched in enumerate(dataloader):# tqdm(dataloader)
        iiii = i_batch + 1
        step_time = time.time()
        # the two thing are same
        input_channel24_mri_real_imag = sample_batched['input_channel24_mri'].to(device)
        target_channel24_mri_real_imag = sample_batched['target_channel24_mri'].to(device)

        # =====================================
        with torch.no_grad():
            reconstruct_4_real_imag = net(input_channel24_mri_real_imag)

        pred_12_real_imag = reconstruct_4_real_imag.detach().permute(0, 2, 3, 1).squeeze()
        input_channel24_real_imag = input_channel24_mri_real_imag.detach().permute(0, 2, 3, 1).squeeze()
        target_channel24_mri_real_imag = target_channel24_mri_real_imag.detach().permute(0, 2, 3, 1).squeeze()

        pred_12_real_imag = pred_12_real_imag.cpu().numpy()
        input_channel24_real_imag = input_channel24_real_imag.cpu().numpy()
        target_channel24_mri_real_imag = target_channel24_mri_real_imag.cpu().numpy()

        # input
        input_12coil_complex = np.zeros([256, 256, channel_ONE], dtype=np.complex64)
        for i in range(0, channel_in, 2):
            input_12coil_complex[:, :, int(i / 2)] = input_channel24_real_imag[:, :,
                                                     i] + 1j * input_channel24_real_imag[:, :, i + 1]
        # input: from kspace to image
        input_12coil_complex_img = np.zeros(input_12coil_complex.shape, dtype=np.complex64)
        for i in range(input_12coil_complex.shape[-1]):
            input_12coil_complex_img[:, :, i] = np.fft.ifft2(input_12coil_complex[:, :, i])


        path_root = f'./results_{channel_ONE}to{channel_TWO}'
        path_list = [f'{path_root}/input', f'{path_root}/ori', f'{path_root}/for']
        for kk in path_list:
            if not os.path.isdir(kk):
                os.makedirs(kk)

        savemat(f'{path_list[0]}/{iiii}.mat', {'Img': input_12coil_complex_img})
        # input_12coil_complex_sos = np.sqrt(np.sum(np.abs((input_12coil_complex_img) ** 2), axis=2))
        # ori
        ori_complex = np.zeros([256, 256, channel_ONE], dtype=np.complex64)
        for i in range(0, channel_in, 2):
            ori_complex[:, :, int(i / 2)] = target_channel24_mri_real_imag[:, :,
                                            i] + 1j * target_channel24_mri_real_imag[:, :, i + 1]
        # kspace to image
        ori_complex_img = np.zeros(ori_complex.shape, dtype=np.complex64)
        for i in range(ori_complex.shape[-1]):
            ori_complex_img[:, :, i] = np.fft.ifft2(ori_complex[:, :, i])
        savemat(f'{path_list[1]}/{iiii}.mat', {'Img': ori_complex_img})
        ori_complex_sos = np.sqrt(np.sum(np.abs((ori_complex_img) ** 2), axis=2))

        # for
        pred_12_complex = np.zeros([256, 256, channel_ONE], dtype=np.complex64)
        for i in range(0, channel_in, 2):
            pred_12_complex[:, :, int(i / 2)] = pred_12_real_imag[:, :, i] + 1j * pred_12_real_imag[:, :, i + 1]
        # kspace to image
        pred_12_complex_img = np.zeros(pred_12_complex.shape, dtype=np.complex64)
        for i in range(pred_12_complex.shape[-1]):
            pred_12_complex_img[:, :, i] = np.fft.ifft2(pred_12_complex[:, :, i])
        savemat(f'{path_list[2]}/{iiii}.mat', {'Img': pred_12_complex_img})


        pred_12_complex_sos = 0
        for kk in range(bs):
            pred_12_complex_img_1 = np.zeros([256, 256, channel_TWO], dtype=np.complex64)
            pred_12_complex_img_1 = pred_12_complex_img[:, :, channel_TWO*kk : channel_TWO*(kk+1)]
            pred_12_complex_sos_1 = np.sqrt(np.sum(np.abs((pred_12_complex_img_1) ** 2), axis=2))
            pred_12_complex_sos = pred_12_complex_sos + pred_12_complex_sos_1
        pred_12_complex_sos = pred_12_complex_sos / bs

        # plt.subplot(1 ,3, 1)
        # plt.imshow(255*abs(np.rot90(ori_complex_sos, -1)) ,cmap='gray')
        # plt.title("ori_sos")
        # plt.subplot(1 ,3, 2)
        # #plt.imshow(255*abs(np.rot90(input_12coil_complex_sos, -1)) ,cmap='gray')
        # plt.imshow(255*abs(np.rot90(ori_complex_sos, -1)) ,cmap='gray')
        # plt.title("input_sos")
        # plt.subplot(1 ,3, 3)
        # plt.imshow(255*abs(np.rot90(np.real(pred_12_complex_sos), -1)) ,cmap='gray')
        # plt.title("pred_sos")
        # plt.show()
        psnr_compress = compare_psnr(255 * abs(ori_complex_sos), 255 * abs(pred_12_complex_sos), data_range=255)
        ssim_compress = compare_ssim(abs(ori_complex_sos), abs(pred_12_complex_sos), data_range=1)
        # print('psnr_forward:',psnr_compress,'    ssim_forward:',ssim_compress)
        PSNR_COMPRESS.append(psnr_compress)
        SSIM_COMPRESS.append(ssim_compress)

        del reconstruct_4_real_imag
    ave_psnr_compress = sum(PSNR_COMPRESS) / len(PSNR_COMPRESS)
    ave_ssim_compress = sum(SSIM_COMPRESS) / len(SSIM_COMPRESS)
    print("ave_psnr_forward: %.10f || ave_ssim_forward:%.10f" % (ave_psnr_compress, ave_ssim_compress))

    path_dir = f"./result_all/"
    path_best_step = f"{path_dir}/all.csv"

    # delete all step when get a new start
    if os.path.exists(path_best_step) and firstTime==1:
        firstTime = 0
        os.remove(path_best_step)
        print("Delete old files succeed ... Now it is a New start....")

    WriteInfo_zl(path_best_step,
                 from_channel=channel_ONE, end_channnel=channel_TWO,
                 PSNR=ave_psnr_compress, SSIM=ave_ssim_compress,modelNum = modelNum)




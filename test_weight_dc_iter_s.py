import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from model.model import InvISPNet
# from dataset.mri_dataset import mriDataset, mriDataset12, mriDataset12_real_imag, mriDataset12_real_imag_cross
from dataset.mri_dataset_weight import mriDataset_real_imag_cross
from config.config import get_arguments
from utils.commons import denorm, preprocess_test_patch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from scipy.io import savemat, loadmat
import os.path as osp
import cv2
import scipy.io as io
from matplotlib import pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
channel_ONE = 2
channel_TWO = 2
bs = channel_ONE // channel_TWO
channel_in = int(channel_ONE * 2)
channel_last = int(channel_TWO * 2)
modelNum = 'latest'

parser = get_arguments()
parser.add_argument("--ckpt_10", type=str, default=f'./exps/{channel_ONE}to{channel_TWO}/checkpoint/{modelNum}.pth',
                    help="Checkpoint path.")
parser.add_argument("--ckpt_20", type=str, default=f'./exps/{channel_ONE}to{channel_TWO}/checkpoint/{modelNum}.pth',
                    help="Checkpoint path.")
parser.add_argument("--ckpt_40", type=str, default=f'./exps/{channel_ONE}to{channel_TWO}/checkpoint/{modelNum}.pth',
                    help="Checkpoint path.")
parser.add_argument("--ckpt_60", type=str, default=f'./exps/{channel_ONE}to{channel_TWO}/checkpoint/{modelNum}.pth',
                    help="Checkpoint path.")
parser.add_argument("--ckpt_80", type=str, default=f'./exps/{channel_ONE}to{channel_TWO}/checkpoint/{modelNum}.pth',
                    help="Checkpoint path.")
parser.add_argument("--ckpt_100", type=str, default=f'./exps/{channel_ONE}to{channel_TWO}/checkpoint/{modelNum}.pth',
                    help="Checkpoint path.")
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save results. ")
parser.add_argument("--iter_num", type=int, default="1")
parser.add_argument("--task", type=str, default=f"test")
parser.add_argument("--split_to_patch", dest='split_to_patch', action='store_true', help="Test on patch. ")
args = parser.parse_args()

def write_images(x,image_save_path):
    x = np.clip(x *255 , 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)

def wgt2k(X,W,DC):
    Y = np.multiply(X,1./W)
    Y[W == 0] = DC[W == 0]
    return Y

def wgt2k_t(X,W,DC):
    Y = torch.multiply(X,1./W)
    Y[0, :, :][W == 0] = DC[0, :, :][W == 0]
    return Y

def load_m_w(w_path,m_path):

    w = loadmat(w_path)['weight']
    mask = np.zeros([256, 256, 5], dtype=np.float)
    for i in range(5):
        if i ==0:
            index = 10
        else:index = 20*i
        mask[:,:,i] = loadmat(m_path +str(index) +'.mat')['mask']
    return w,mask

def get_next_net(input_channel4_mri_real_imag,reconstruct_4_real_imag,ww,mask1,mask2):
    input_channel4_mri_real_imag = input_channel4_mri_real_imag.cpu().numpy().squeeze()
    reconstruct_4_real_imag = reconstruct_4_real_imag.cpu().numpy().squeeze()

    forward = np.zeros([2, 256, 256], dtype=np.complex64)
    input_k = np.zeros([2, 256, 256], dtype=np.complex64)

    forward_nw = np.zeros([2, 256, 256], dtype=np.complex64)
    k_complex2 = np.zeros([2, 256, 256], dtype=np.complex64)
    k_sampled = np.zeros([2, 256, 256], dtype=np.complex64)
    k_w = np.zeros([2, 256, 256], dtype=np.complex64)
    ##实虚合并
    forward[0, :, :] = reconstruct_4_real_imag[0, :, :]+1j*reconstruct_4_real_imag[1, :, :]
    forward[1, :, :] = reconstruct_4_real_imag[2, :, :]+1j*reconstruct_4_real_imag[3, :, :]
    input_k[0, :, :] = input_channel4_mri_real_imag[0, :, :]+1j*input_channel4_mri_real_imag[1, :, :]
    input_k[1, :, :] = input_channel4_mri_real_imag[2, :, :]+1j*input_channel4_mri_real_imag[3, :, :]
    ##DC
    forward_nw[0,:,:] =wgt2k(forward[0,:,:],ww,input_k[1,:,:])
    forward_nw[1,:,:] = wgt2k(forward[1,:,:],ww,input_k[1,:,:])
    k_complex2[0,:,:] = input_k[1,:,:] + forward_nw[0,:,:] * (1 - mask1)
    k_complex2[1,:,:] = input_k[1,:,:] + forward_nw[1,:,:] * (1 - mask1)

    #average
    k_complex2_av = np.mean(k_complex2,  axis=0)

    #*mask *weight
    k_sampled[0,:, :] =  input_k[0,:, :]
    k_sampled[1, :, :] = np.multiply(k_complex2_av,mask2)
    k_w[0, :, :] = np.multiply(k_sampled[0,:, :], ww)
    k_w[1, :, :] = np.multiply(k_sampled[1,:, :], ww)

    #实虚分离
    img_real_w = np.real(k_w)
    img_real_w = np.array(img_real_w, dtype=np.float32)
    img_imag_w = np.imag(k_w)
    img_imag_w = np.array(img_imag_w, dtype=np.float32)
    img_w = np.zeros([4,256, 256], dtype=np.float32)
    for i in range(0, 4, 2):
        img_w[i,:, :] = img_real_w[int(i / 2),:, :]
        img_w[i+ 1,:, :] = img_imag_w[int(i / 2),:, :]

    img_real_s = np.real(k_sampled)
    img_real_s = np.array(img_real_s, dtype=np.float32)
    img_imag_s = np.imag(k_sampled)
    img_imag_s = np.array(img_imag_s, dtype=np.float32)
    img_s = np.zeros([4, 256, 256], dtype=np.float32)
    for i in range(0, 4, 2):
        img_s[i, :, :] = img_real_s[int(i / 2), :, :]
        img_s[i + 1, :, :] = img_imag_s[int(i / 2), :, :]



    return k_complex2_av,img_w,img_s



ckpt_name = args.ckpt_40.split("/")[-1].split(".")[0]
if args.split_to_patch:
    os.makedirs(args.out_path + "%s/results_metric_%s/" % (args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path + "%s/results_metric_%s/" % (args.task, ckpt_name)
else:
    os.makedirs(args.out_path + "%s/results_%s/" % (args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path + "%s/results_%s/" % (args.task, ckpt_name)


def main(args):
    net_10 = InvISPNet(channel_in=channel_in, channel_out=channel_in, block_num=8)
    device = torch.device("cuda:0")
    net_10.to(device)
    net_10.eval()

    net_20 = InvISPNet(channel_in=channel_in, channel_out=channel_in, block_num=8)
    device = torch.device("cuda:0")
    net_20.to(device)
    net_20.eval()

    net_40 = InvISPNet(channel_in=channel_in, channel_out=channel_in, block_num=8)
    device = torch.device("cuda:0")
    net_40.to(device)
    net_40.eval()

    net_60 = InvISPNet(channel_in=channel_in, channel_out=channel_in, block_num=8)
    device = torch.device("cuda:0")
    net_60.to(device)
    net_60.eval()

    net_80 = InvISPNet(channel_in=channel_in, channel_out=channel_in, block_num=8)
    device = torch.device("cuda:0")
    net_80.to(device)
    net_80.eval()

    net_100 = InvISPNet(channel_in=channel_in, channel_out=channel_in, block_num=8)
    device = torch.device("cuda:0")
    net_100.to(device)
    net_100.eval()

    if os.path.isfile(args.ckpt_10):
        net_10.load_state_dict(torch.load(args.ckpt_10), strict=False)
        print("[INFO] Loaded checkpoint_10: {}".format(args.ckpt_10))

    if os.path.isfile(args.ckpt_20):
        net_20.load_state_dict(torch.load(args.ckpt_20), strict=False)
        print("[INFO] Loaded checkpoint_20: {}".format(args.ckpt_20))

    if os.path.isfile(args.ckpt_40):
        net_40.load_state_dict(torch.load(args.ckpt_40), strict=False)
        print("[INFO] Loaded checkpoint_40: {}".format(args.ckpt_40))

    if os.path.isfile(args.ckpt_60):
        net_60.load_state_dict(torch.load(args.ckpt_60), strict=False)
        print("[INFO] Loaded checkpoint_60: {}".format(args.ckpt_60))

    if os.path.isfile(args.ckpt_80):
        net_80.load_state_dict(torch.load(args.ckpt_80), strict=False)
        print("[INFO] Loaded checkpoint_80: {}".format(args.ckpt_80))

    if os.path.isfile(args.ckpt_100):
        net_100.load_state_dict(torch.load(args.ckpt_100), strict=False)
        print("[INFO] Loaded checkpoint_100: {}".format(args.ckpt_100))

    print("[INFO] Start data load and preprocessing")

    mri_dataset = mriDataset_real_imag_cross(
        # root1='/zw/data/zh/T1_test',
        # root2='/zw/data/zh/T2_test',
        root1='/hy/M4Raw/data/T1_test',
        root2='/hy/M4Raw/data/T2_test',
        # root='/zw/ACNN/result_mask_2/20.mat')
        #  root='/zw/ACNN/result_mask_2/10.mat')
        root = '/zw/data/mask/radial/20.mat')
    dataloader = DataLoader(mri_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    print("[INFO] Start test...")
    PSNR_zero = []
    SSIM_zero = []
    PSNR_COMPRESS = []
    SSIM_COMPRESS = []
    start_time = time.time()
    for i_batch, sample_batched in enumerate(dataloader):
        iiii = i_batch + 1
        print(iiii)
        step_time = time.time()
        iter_num = args.iter_num
        w_path =  '/zw_2/M4Raw/data/weight/weight_T1_0.5_1e2.mat'
        m_path = '/zw/data/mask/radial/'
        # m_path = '/zw/data/mask/radial/'
        weight,mask = load_m_w(w_path,m_path)

        # the two thing are same
        # +weight
        input_channel4_mri_real_imag = sample_batched['input_channel4_T12_mri_w'].to(device)
        # ori data
        input = sample_batched['input_channel4_T12_mri'].to(device)
        target = sample_batched['target_channel4_T2_mri'].to(device)
        mask_dc = sample_batched['mask_dc']
        mask_dc = mask_dc.cpu().numpy().squeeze()
        input_t = input.detach().permute(0, 3, 1, 2)
        # #
        # input_channel4_mri_real_imag[:,0,:,:] = input_channel4_mri_real_imag[:,2,:,:]
        # input_channel4_mri_real_imag[:, 1, :, :] = input_channel4_mri_real_imag[:,3,:,:]

        # with torch.no_grad():
        #     reconstruct_4_real_imag = net_10(input_channel4_mri_real_imag)
        #
        # reconstruct_10,input_w_next,input_nw_next= get_next_net(input_t,
        #                                                         reconstruct_4_real_imag,
        #                                                         weight,mask_dc,mask[:,:,2])
        # input_w_next = torch.Tensor(input_w_next).unsqueeze(0).to(device)
        # input_nw_next = torch.Tensor(input_nw_next).unsqueeze(0).to(device)
        # reconstruct = reconstruct_10
        # # #
        # input_w_next[:, 0, :, :] = input_w_next[:, 2, :, :]
        # input_w_next[:, 1, :, :] = input_w_next[:,3,:,:]
        # #
        assert 0
        with torch.no_grad():
            reconstruct_4_real_imag = net_20(input_channel4_mri_real_imag)
        #
        reconstruct_20,input_w_next,input_nw_next= get_next_net(input_t,
                                                                reconstruct_4_real_imag,
                                                                weight,mask_dc,mask[:,:,2])
        input_w_next = torch.Tensor(input_w_next).unsqueeze(0).to(device)
        input_nw_next = torch.Tensor(input_nw_next).unsqueeze(0).to(device)
        reconstruct = reconstruct_20
        # # # # #
        # input_w_next[:,0,:,:] = input_w_next[:,2,:,:]
        # input_w_next[:, 1, :, :] = input_w_next[:,3,:,:]
        # #
        with torch.no_grad():
            reconstruct_4_real_imag = net_40(input_w_next)

        reconstruct_40,input_w_next,input_nw_next= get_next_net(input_t,
                                                                reconstruct_4_real_imag,
                                                                weight,mask_dc,mask[:,:,3])
        input_w_next = torch.Tensor(input_w_next).unsqueeze(0).to(device)
        input_nw_next = torch.Tensor(input_nw_next).unsqueeze(0).to(device)
        reconstruct = reconstruct_40
        # # # # # # # # # # # # #
        # input_w_next[:, 0, :, :] = input_w_next[:, 2, :, :]
        # input_w_next[:, 1, :, :] = input_w_next[:, 3, :, :]
        # # # # # #
        # with torch.no_grad():
        #     reconstruct_4_real_imag = net_60(input_w_next)
        # 
        # reconstruct_60, input_w_next,input_nw_next = get_next_net(input_t,
        #                                                      reconstruct_4_real_imag,
        #                                                      weight, mask_dc,mask[:,:,4])
        # input_w_next = torch.Tensor(input_w_next).unsqueeze(0).to(device)
        # input_nw_next = torch.Tensor(input_nw_next).unsqueeze(0).to(device)
        # reconstruct = reconstruct_60
        # # # # # # # # # # # # # # #
        # input_w_next[:, 0, :, :] = input_w_next[:, 2, :, :]
        # input_w_next[:, 1, :, :] = input_w_next[:, 3, :, :]
        # # # # # # # # # # #
        # with torch.no_grad():
        #     reconstruct_4_real_imag = net_80(input_w_next)
        #
        # reconstruct_80, input_w_next,input_nw_next = get_next_net(input_t,
        #                                                      reconstruct_4_real_imag,
        #                                                      weight, mask_dc, mask[:,:,4])
        # input_w_next = torch.Tensor(input_w_next).unsqueeze(0).to(device)
        # input_nw_next = torch.Tensor(input_nw_next).unsqueeze(0).to(device)
        # reconstruct = reconstruct_80
        # # # # #
        # # # # input_w_next[:, 0, :, :] = input_w_next[:, 2, :, :]
        # # # # input_w_next[:, 1, :, :] = input_w_next[:, 3, :, :]
        # # #
        # with torch.no_grad():
        #     reconstruct_4_real_imag = net_100(input_w_next)
        #
        # reconstruct_100, input_w_next,input_nw_next = get_next_net(input_t,
        #                                                      reconstruct_4_real_imag,
        #                                                      weight, mask_dc, mask_100)
        # reconstruct = reconstruct_100
        # # #


        target = target.cpu().numpy().squeeze()
        input = input.cpu().numpy().squeeze()
        #
        # input
        input_2coil_complex = np.zeros([256, 256, channel_ONE], dtype=np.complex64)
        for i in range(0, channel_in, 2):
            input_2coil_complex[:, :, int(i / 2)] = input[:, :,i] + 1j * input[:, :, i + 1]

        input_2coil_complex_img = np.zeros(input_2coil_complex.shape, dtype=np.complex64)
        for i in range(input_2coil_complex.shape[-1]):
            input_2coil_complex_img[:, :, i] = np.fft.ifft2(np.fft.ifftshift(input_2coil_complex[:, :, i]))
        input_2coil_complex_img_sos = input_2coil_complex_img[:, :, 1]

        path_root = f'./results_{channel_ONE}to{channel_TWO}'
        path_list = [f'{path_root}/input', f'{path_root}/ori', f'{path_root}/for']
        for kk in path_list:
            if not os.path.isdir(kk):
                os.makedirs(kk)
        savemat(f'{path_list[0]}/{iiii}.mat', {'Img': input_2coil_complex_img})

        # get T2 sampled
        ori_complex = np.zeros([256, 256, channel_ONE], dtype=np.complex64)
        for i in range(0, channel_in, 2):
            ori_complex[:, :, int(i / 2)] = target[:, :,i] + 1j * target[:, :, i + 1]



        # kspace to image
        ori_complex_img = np.zeros(ori_complex.shape, dtype=np.complex64)
        for i in range(ori_complex.shape[-1]):
            ori_complex_img[:, :, i] = np.fft.ifft2(np.fft.ifftshift(ori_complex[:, :, i]))
        savemat(f'{path_list[1]}/{iiii}.mat', {'Img': ori_complex_img})

        # target average
        ori_complex_sos = np.mean(ori_complex_img, axis=2)

        # forward
        # kspace to image
        pred_2_complex_img = np.fft.ifft2(np.fft.ifftshift(reconstruct))
        savemat(f'{path_list[2]}/{iiii}.mat', {'Img': pred_2_complex_img})
        pred_2_complex_sos = pred_2_complex_img


        ori_complex_sos = ori_complex_sos / np.max(abs(ori_complex_sos))
        pred_2_complex_sos = pred_2_complex_sos / np.max(abs(pred_2_complex_sos))
        input_2coil_complex_img_sos = input_2coil_complex_img_sos / np.max(abs(input_2coil_complex_img_sos))
        ## zero_psnr and ssim
        psnr_compress_zero = compare_psnr(255 * abs(ori_complex_sos), 255 * abs(input_2coil_complex_img_sos), data_range=255)
        ssim_compress_zero = compare_ssim(abs(ori_complex_sos), abs(input_2coil_complex_img_sos), data_range=1)


        psnr_compress = compare_psnr(255 * abs(ori_complex_sos), 255 * abs(pred_2_complex_sos), data_range=255)
        ssim_compress = compare_ssim(abs(ori_complex_sos), abs(pred_2_complex_sos), data_range=1)

        write_images(abs(ori_complex_sos), osp.join('./results/' + 'target' + '.png'))
        write_images(abs(input_2coil_complex_img_sos), osp.join('./results/' + 'input' + '.png'))
        write_images(abs(pred_2_complex_sos), osp.join('./results/' + 'pred' + '.png'))

        # print('psnr_zero:', psnr_compress_zero, '    ssim_zero:', ssim_compress_zero)
        # print('psnr_forward:',psnr_compress,'    ssim_forward:',ssim_compress)

        PSNR_COMPRESS.append(psnr_compress)
        SSIM_COMPRESS.append(ssim_compress)
        PSNR_zero.append(psnr_compress_zero)
        SSIM_zero.append(ssim_compress_zero)


        # print("[INFO] step time: ", time.time() - step_time)
        del reconstruct_4_real_imag
    ave_psnr_compress = sum(PSNR_COMPRESS) / len(PSNR_COMPRESS)
    ave_ssim_compress = sum(SSIM_COMPRESS) / len(SSIM_COMPRESS)
    ave_psnr_zero = sum(PSNR_zero) / len(PSNR_zero)
    ave_ssim_zero = sum(SSIM_zero) / len(SSIM_zero)
    print("ave_psnr_forward: %.10f || ave_ssim_forward:%.10f" % (ave_psnr_compress, ave_ssim_compress))
    print("ave_psnr_zero: %.10f || ave_ssim_zero:%.10f" % (ave_psnr_zero, ave_ssim_zero))
    # print("[INFO] rec time: ", time.time() - start_time)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main(args)

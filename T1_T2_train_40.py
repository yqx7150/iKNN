import numpy as np
import os, time, random
import argparse
import json
import torch.nn.functional as F
import torch
import cv2
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from model.model import InvISPNet
from dataset.mri_dataset_weight import mriDataset_real_imag_cross
from config.config import get_arguments
from skimage.measure import compare_psnr, compare_ssim
from scipy.io import savemat, loadmat
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
channel_ONE = 2
channel_TWO = 2
bs = channel_ONE // channel_TWO
channel_in = int(channel_ONE * 2)
channel_last = int(channel_TWO * 2)

parser = get_arguments()

parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save checkpoint. ")
parser.add_argument("--task", type=str, default=f"{channel_ONE}to{channel_TWO}", help="tasktasktasktask")
parser.add_argument("--gamma", dest='gamma', action='store_true',
                    help="Use gamma compression for raw data.(zl think it is useless)", default=True)
parser.add_argument("--resume", default=False, dest='resume', action='store_true', help="Resume training. ")
parser.add_argument("--loss", type=str, default="L1", choices=["L1", "L2"], help="Choose which loss function to use. ")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--epoch", type=int, default=5000, help="Learning epuch")
parser.add_argument("--aug", dest='aug', action='store_true', help="Use data augmentation.")
args = parser.parse_args()

os.makedirs(args.out_path, exist_ok=True)
os.makedirs(args.out_path + "%s" % args.task, exist_ok=True)
os.makedirs(args.out_path + "%s/checkpoint" % args.task, exist_ok=True)

with open(args.out_path + "%s/commandline_args.yaml" % args.task, 'w') as f:
    json.dump(args.__dict__, f, indent=2)


def sos_torch_auto(img_tensor):
    shape = img_tensor.size()
    tensor_square = torch.mul(img_tensor, img_tensor)
    tensor_sum = torch.FloatTensor(shape[0], int(shape[1] / 2), shape[2], shape[3])
    for i in range(0, int(2 * tensor_sum.size()[1]), 2):
        tensor_sum[:, int(i / 2), :, :] = tensor_square[:, i, :, :] + tensor_square[:, i + 1, :, :]
    tensor_sum_all = torch.sum(tensor_sum, dim=1)
    tensor_sqrt = torch.sqrt(tensor_sum_all)
    tensor_sqrt = tensor_sqrt.cuda()

    return tensor_sqrt


def write_images(x, image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)


# 复数聚合 k空间
def get_kdata(tensor):
    tensor_temp = torch.ones(
        [1, tensor.size()[1] // 2, tensor.size()[2], tensor.size()[3]]).cuda()
    tensor_kdata = torch.complex(tensor_temp, tensor_temp)
    for i in range(0, tensor.size()[1], 2):
        tensor_kdata[0, int(i / 2), :, :] = torch.complex(tensor[0, i, :, :],
                                                          tensor[0, i + 1, :, :])
    return tensor_kdata


# 复数聚合 图像域
def get_idata(tensor, tensor2):
    tensor_temp = torch.ones(
        [1, tensor2.size()[1] // 2, tensor2.size()[2], tensor2.size()[3]]).cuda()
    tensor_img = torch.complex(tensor_temp, tensor_temp)
    for i in range(tensor2.size()[1] // 2):
        tensor_img[0, i, :, :] = torch.fft.ifft2(torch.fft.ifftshift(tensor[0, i, :, :]))

    return tensor_img


# 图像域复数分离
def get_separation(tensor, tensor2):
    tensor_img_separation = torch.FloatTensor(1, tensor2.size()[1], tensor2.size()[2],
                                              tensor2.size()[3]).cuda()

    for i in range(0, tensor2.size()[1], 2):
        tensor_img_separation[0, i, :, :] = torch.real(tensor[0, int(i / 2), :, :])
        tensor_img_separation[0, i + 1, :, :] = torch.imag(tensor[0, int(i / 2), :, :])

    return tensor_img_separation


def write_Data(model_num, psnr, ssim, path):
    filedir = "result.txt"
    with open(osp.join(path, filedir), "w+") as f:  # a+
        f.writelines(str(model_num) + ' ' + '[' + str(psnr) + ' ' + str(ssim) + ']')
        f.write('\n')


def write_Data2(psnr, ssim, path):
    filedir = "result.txt"
    with open(osp.join(path, filedir), "a+") as f:  # a+
        f.writelines('[' + str(psnr) + ' ' + str(ssim) + ']')
        f.write('\n')


def main(args):
    # 设置网络
    def sos_torch_st(img_tensor, offset):
        tensor_square = torch.mul(img_tensor, img_tensor)
        tensor_sum = torch.FloatTensor(1, channel_TWO, 256, 256).cuda()
        for i in range(0, int(2 * tensor_sum.size()[1]), 2):
            tensor_sum[:, int(i / 2), :, :] = torch.add(tensor_square[:, i + int(offset), :, :],
                                                        tensor_square[:, i + 1 + int(offset), :, :])
        tensor_sum_all = torch.sum(tensor_sum, dim=1)
        tensor_sqrt = torch.sqrt(tensor_sum_all)
        tensor_sqrt = tensor_sqrt.cuda()

        return tensor_sqrt

    # 修改通道请改此处
    net = InvISPNet(channel_in=channel_in, channel_out=channel_in, block_num=8)
    # 将网络加载到cuda上
    net.cuda()
    # 如果有之前训练了的模型，可以先加载已保存的网络权重，之后再训练
    if args.resume:
        net.load_state_dict(torch.load(args.out_path + "%s/checkpoint/latest.pth" % args.task))
        print("[INFO] loaded " + args.out_path + "%s/checkpoint/latest.pth" % args.task)
    # 设置优化器，这里可以设置成其它优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[(i + 1) * 25 for i in range((args.epoch) // 25 - 1)],
                                         gamma=0.5)
    print("[INFO] Start data loading and preprocessing")
    # 数据集加载

    # root1是输入，root是输入，root2标签，但已经积重难返，还是不删为好。
    mri_dataset = mriDataset_real_imag_cross(
        # root1='/zw/data/Train/T1_T2sampled',
        # root2='/zw/data/Train/T2_copy',
        # root='/zw/data/Train/T1_T2sampled')
        # '/hy/M4Raw/data/T1_train'
        # root1='/zw/data/zh/T1',
        # root2='/zw/data/zh/T2',
        root1='/hy/M4Raw/data/T1_train',
        root2='/hy/M4Raw/data/T2_train',
        # "/zw/data/mask/radial/20.mat"
        # root="/zw/ACNN/result_mask_2/40.mat")
        # root='/zw/data/mask/radial/40.mat')
        root='/zw/data/mask/Uniform/Uniform1D_R3_24.mat')
    # root1='/zw/data_sait/T1_train',
    # root1='/zw/data_sait/T1_train',
    # root2='/zw/data_sait/T2_train',
    # root='/zw/data/mask/radial/40.mat')
    dataloader = DataLoader(mri_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    print("[INFO] Start to train")
    step = 0
    for epoch in range(args.epoch):
        epoch_time = time.time()
        # train
        i = 0
        for i_batch, sample_batched in enumerate(dataloader):
            step_time = time.time()
            input_T12_mri, target_T2_mri = sample_batched['input_channel4_T12_mri_w'].cuda(), \
                                           sample_batched['target_channel4_T2_mri_w'].cuda()
            # 将输入进入网络中
            # input_T12_mri = input_T12_mri / torch.max(abs(input_T12_mri))
            # target_T2_mri = target_T2_mri / torch.max(abs(target_T2_mri))
            # input_T12_mri[:, 0, :, :] = input_T12_mri[:, 2, :, :]
            # input_T12_mri[:, 1, :, :] = input_T12_mri[:, 3, :, :]
            forward_compress = net(input_T12_mri)
            rev_compress = net(forward_compress, rev=True)

            # 网络正向输出
            forward_kdata = get_kdata(forward_compress)
            forward_idata = get_idata(forward_kdata, forward_compress)
            # forward_idata_separation = get_separation(forward_idata,forward_compress)
            forward_compress_av = torch.FloatTensor(1, 2, 256, 256).cuda()
            forward_compress_av[:, 0, :, :] = ((forward_compress[:, 0, :, :] + forward_compress[:, 2, :, :])/2)
            forward_compress_av[:, 1, :, :] = ((forward_compress[:, 1, :, :] + forward_compress[:, 3, :, :])/2)
            # #网络逆向输出
            # rev_kdata = get_kdata(rev_compress)
            # rev_idata = get_idata(rev_kdata,rev_compress)
            # #输入
            input_kdata = get_kdata(input_T12_mri)
            input_idata = get_idata(input_kdata, input_T12_mri)
            # #标签
            # target_kdata = get_kdata(target_T2_mri)
            # target_idata = get_idata(target_kdata, target_T2_mri)
            # target_idata_separation = get_separation(target_idata,target_T2_mri)
            target_T2_mri_av = torch.FloatTensor(1, 2, 256, 256).cuda()
            target_T2_mri_av[:, 0, :, :] = ((target_T2_mri[:, 0, :, :] + target_T2_mri[:, 2, :, :])/2)
            target_T2_mri_av[:, 1, :, :] =( (target_T2_mri[:, 1, :, :] + target_T2_mri[:, 3, :, :])/2)

            forward_loss_k = F.smooth_l1_loss(forward_compress_av, target_T2_mri_av)
            rev_loss_k = F.smooth_l1_loss(rev_compress, input_T12_mri)

            loss = args.forward_weight * forward_loss_k + rev_loss_k
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pic_path = './pic/ww/30/'
            # T1 = input_idata[0, 0, :, :]
            # T2 = input_idata[0, 1, :, :]
            # T1 = T1.detach().cpu().numpy()
            # T2 = T2.detach().cpu().numpy()
            # T1 = T1 / np.max(abs(T1))
            # T2 = T2 / np.max(abs(T2))
            # write_images(abs(T1), osp.join(pic_path + 'T1' + '.png'))
            # write_images(abs(T2), osp.join(pic_path + 'T2' + '.png'))

            # #
            # T3 = target_idata[0, 0, :, :]
            # T4 = target_idata[0, 1, :, :]
            # T3 = T3.detach().cpu().numpy()
            # T4 = T4.detach().cpu().numpy()
            # T3 = T3 / np.max(abs(T3))
            # T4 = T4 / np.max(abs(T4))
            # write_images(abs(T3), osp.join(pic_path + 'T3' + '.png'))
            # write_images(abs(T4), osp.join(pic_path + 'T4' + '.png'))

            # T5 = forward_idata[0, 0, :, :]
            # T6 = forward_idata[0, 1, :, :]
            # T5 = T5.detach().cpu().numpy()
            # T6 = T6.detach().cpu().numpy()
            # # T3 = T3.detach().cpu().numpy()
            # T5 = T5 / np.max(abs(T5))
            # T6 = T6 / np.max(abs(T6))
            # # T3 = T3 / np.max(abs(T3))
            # write_images(abs(T5), osp.join(pic_path + 'T5' + '.png'))
            # write_images(abs(T6), osp.join(pic_path + 'T6' + '.png'))

            write_Data2(forward_loss_k.detach().cpu().numpy(), rev_loss_k.detach().cpu().numpy(), pic_path)
            if step % 100 == 0:
                print(
                    "Epoch: %d Step: %d || loss: %.10f forward_4k_loss:  %.10f rev_4k_loss:  %.10f|| lr: %f time: %f" % (
                        epoch, step, loss.detach().cpu().numpy(), forward_loss_k.detach().cpu().numpy(),
                        rev_loss_k.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time() - step_time))

            step += 1
            i = i + 1
        torch.save(net.state_dict(), args.out_path + "%s/checkpoint/latest.pth" % args.task)
        if (epoch + 1) % 1 == 0:
            torch.save(net.state_dict(), args.out_path + "%s/checkpoint/%04d.pth" % (args.task, epoch))
            print("[INFO] Successfully saved " + args.out_path + "%s/checkpoint/%04d.pth" % (args.task, epoch))
        scheduler.step()
        print("[INFO] Epoch time: ", time.time() - epoch_time, "task: ", args.task)


if __name__ == '__main__':
    # torch.set_num_threads(4)
    main(args)

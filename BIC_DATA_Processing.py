import numpy as np
import scipy.io as io
import  cv2
import os.path as osp


def write_images(x,image_save_path):
    x = np.clip(x *255 , 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)

file_path='/zw/data/target_T2_2ch/5.mat'
ori_data = np.zeros([256,256],dtype=np.complex64)
ori_data = io.loadmat(file_path)['Img']
k_data = ori_data[:,:,1]
k_data = k_data/np.max(abs(k_data))
write_images(abs(k_data),osp.join('./results/'+'Rec'+'.png'))
# k_data = np.fft.fft2(ori_data[:, :])
# i_data = np.fft.ifft2(k_data[:, :])
# print(i_data.dtype)
# i_data = i_data/np.max(abs(i_data))
# io.savemat(osp.join('./results/'+'pro.mat'),{'pro':i_data})


# write_images(abs(i_data),osp.join('./results/'+'ori'+'.png'))
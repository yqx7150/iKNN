from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
images_temp = loadmat('/home/lqg/文档/duibi/data_brain/train_data/train_12coil_zengqiang/train_12ch/1.mat')['Img']
mri_images = np.zeros(images_temp.shape, dtype=np.complex128)
for i in range(mri_images.shape[-1]):
    mri_images[:, :, i] = np.fft.fftshift(np.fft.fft2(images_temp[:, :, i]))
plt.figure(1)
plt.imshow(np.log10(abs(mri_images[:, :, 0])+1), cmap='gray')
plt.show()

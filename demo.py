from scipy import io
import numpy as np

mat = np.load('/zw/data/data_26.npy')

io.savemat('/zw/data/xgy.mat', {'DATA': mat})
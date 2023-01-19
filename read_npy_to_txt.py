# -*- coding: utf-8 -*-
"""
@Time: 2021/9/18 14:58
@Author: Origami_Shen_Li
@FileName: read_npy_to_txt
@Software: Pycharm

@Dedicated to: Who I love and who loves me
"""

import os
import sys

import numpy as np

anomaly = np.load("./results/SWaT/mse_drop0.npy")

print("------type-------")
print(type(anomaly))
print("------shape-------")
print(anomaly.shape)
print("------data-------")
# print(pre_train)
# np.savetxt(f'./data/MSL/MSL_train.csv', anomaly, fmt='%f', delimiter=',')

# np.savetxt("./comparison/SMD/anomaly.txt", anomaly, fmt='%f', delimiter='\n')

# gt = np.load("./results/SMD/gt.npy")
#
# np.savetxt("./comparison/SMD/gt.txt", gt, fmt='%f', delimiter='\n')
#
# gt = np.load("./results/SMD/pred.npy")
#
# np.savetxt("./comparison/SMD/pred.txt", gt, fmt='%f', delimiter='\n')
#
# gt = np.load("./results/SMD/pred_inv.npy")
#
# np.savetxt("./comparison/SMD/pred_inv.txt", gt, fmt='%f', delimiter='\n')
#
# gt = np.load("./results/SMD/true.npy")
#
# np.savetxt("./comparison/SMD/true.txt", gt, fmt='%f', delimiter='\n')
#
# gt = np.load("./results/SMD/true_inv.npy")
#
# np.savetxt("./comparison/SMD/true_inv.txt", gt, fmt='%f', delimiter='\n')

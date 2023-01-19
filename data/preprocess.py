import pandas as pd
import numpy as np
import os

# SMD: SMD does not need any change due to npy format

# MSL: MSL does not need any change due to npy format

# SMAP: SMAP does not need any change due to npy format

# SWAT
Normal_SWAT = pd.read_excel('./SWAT/SWaT_Dataset_Normal_v1.xlsx', header=1)
Normal_SWAT = Normal_SWAT.values[0:, 1:-1].astype(float)  # value
index_Normal_SWAT = np.linspace(1.0, Normal_SWAT.shape[1], num=Normal_SWAT.shape[1], dtype=float)  # index
P_Normal_SWAT = np.concatenate([index_Normal_SWAT.reshape(1, -1), Normal_SWAT], axis=0)
CSV_Normal_SWAT = pd.DataFrame(P_Normal_SWAT)
CSV_Normal_SWAT.to_csv('./SWAT/Normal.csv', index=False, header=False)

Attack_SWAT = pd.read_excel('./SWAT/SWaT_Dataset_Attack_v0.xlsx', header=1)
Attack_SWAT_value = Attack_SWAT.values[0:, 1:-1].astype(float)  # value
index_Normal_SWAT = np.linspace(1.0, Attack_SWAT_value.shape[1] + 1,
                                num=Attack_SWAT_value.shape[1] + 1, dtype=float)  # index
label_SWAT = Attack_SWAT.values[0:, -1]  # label
num_label_SWAT = np.zeros_like(label_SWAT)
num_label_SWAT[np.where(label_SWAT == 'Attack')] = 1
P_Attack_SWAT = np.concatenate([Attack_SWAT_value, num_label_SWAT.reshape(-1, 1)], axis=1)
P_Attack_SWAT = np.concatenate([index_Normal_SWAT.reshape(1, -1), P_Attack_SWAT], axis=0)
CSV_Attack_SWAT = pd.DataFrame(P_Attack_SWAT)
CSV_Attack_SWAT.to_csv('./SWAT/Attack.csv', index=False, header=False)

# PSM: PSM does not need any change due to no discrete variate

# WADI
Normal_WADI = pd.read_csv('./WADI/WADI_14days_new.csv', header=0)
Normal_WADI_part1 = Normal_WADI.values[0:, 3:50].astype(float)  # value 47c ignoring blank variates
Normal_WADI_part2 = Normal_WADI.values[0:, 52:86].astype(float)  # value 34c
Normal_WADI_part3 = Normal_WADI.values[0:, 88:130].astype(float)  # value 42c
Normal_WADI = np.concatenate([Normal_WADI_part1, Normal_WADI_part2, Normal_WADI_part3], axis=1) # value 123c
index_Normal_WADI = np.linspace(1.0, Normal_WADI.shape[1], num=Normal_WADI.shape[1], dtype=float)  # index
P_Normal_WADI = np.concatenate([index_Normal_WADI.reshape(1, -1), Normal_WADI], axis=0)
CSV_Normal_WADI = pd.DataFrame(P_Normal_WADI)
CSV_Normal_WADI.to_csv('./WADI/Normal.csv', index=False, header=False)

Attack_WADI = pd.read_csv('./WADI/WADI_attackdataLABLE.csv', header=1)
Attack_WADI_part1 = Attack_WADI.values[0:, 3:50].astype(float)  # value 47c ignoring blank variates
Attack_WADI_part2 = Attack_WADI.values[0:, 52:86].astype(float)  # value 34c
Attack_WADI_part3 = Attack_WADI.values[0:, 88:130].astype(float)  # value 42c
WADI_label = Attack_WADI.values[0:, 130].astype(float).reshape(-1, 1)  # label
Attack_WADI_label = np.zeros_like(WADI_label)
Attack_WADI_label[np.where(WADI_label == -1)] = 1
Attack_WADI = np.concatenate([Attack_WADI_part1, Attack_WADI_part2, Attack_WADI_part3, Attack_WADI_label],
                             axis=1)  # value 123c + 1 label
index_Attack_WADI = np.linspace(1.0, Attack_WADI.shape[1], num=Attack_WADI.shape[1], dtype=float)  # index
P_Attack_WADI = np.concatenate([index_Attack_WADI.reshape(1, -1), Attack_WADI], axis=0)
CSV_Attack_WADI = pd.DataFrame(P_Attack_WADI)
CSV_Attack_WADI.to_csv('./WADI/Attack.csv', index=False, header=False)

# MBA: MBA does not need any change due to no discrete variate

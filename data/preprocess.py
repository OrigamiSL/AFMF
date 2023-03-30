import pandas as pd
import numpy as np
import os
import time

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

# MBA: MBA does not need any change

# UCR: UCR does not need any change

# NAB: NAB does not need any change

# MSDS
logs = pd.read_csv('./MSDS/concurrent_data/logs/logs_aggregated_concurrent.csv')
log = logs[['Timestamp'] + ['Hostname'] + ['log_level']]
label = np.argwhere(list(log['log_level'] == 'ERROR'))
error = log.iloc[label[:,0]]
label = np.argwhere(list(error['Hostname'] == 'wally113'))
error = error.iloc[label[:,0]]

anomaly = error['Timestamp']
error = error.reset_index(drop=True)
file_path = './MSDS/concurrent_data/metrics/'
file_name = 'wally113_metrics_concurrent.csv'
labels = []
df = pd.read_csv(file_path + file_name)
df = df.drop(columns=['load.cpucore', 'load.min1', 'load.min5', 'load.min15'])
id_vars = ['now']

melted = df.melt(id_vars=id_vars).dropna()
df = melted.pivot_table(index=id_vars, columns="variable", values="value")
df_merged = df
error['Timestamp'] = pd.DatetimeIndex(error['Timestamp']).strftime("%Y-%m-%d %H:%M:%S")
error = error.drop(np.argwhere(list(error['Timestamp'] < '2019-11-25 15:58:58')).reshape(-1))
error = error.sort_values(by='Timestamp')
error = error.reset_index(drop=True)
labels = np.zeros(len(df_merged))

i = 0
for ind, timestemp in enumerate(error['Timestamp'].values):
    while i < len(df_merged):
        if timestemp in df_merged.index[i]:
            labels[max(0, i-4):min(len(df_merged), (i+4))] = 1
            break
        i += 1

value = df_merged.values
train_data = []
test_data = []
for i in range(len(df_merged)):
    times = df_merged.index[i].split(' ')
    timestemps = time.strptime(times[1], "%H:%M:%S")
    init_time = time.strptime('20:00:00', "%H:%M:%S")
    if times[0] == '2019-11-25' and timestemps < init_time:
        test_data.append(value[i])
    else:
        train_data.append(value[i])
labels = labels[:len(test_data)]
train_data = np.array(train_data)
test_data = np.array(test_data)
np.savetxt('./MSDS/labels.csv', labels, fmt='%.f', delimiter=',')
np.savetxt('./MSDS/train.csv', train_data, fmt='%.6f', delimiter=',')
np.savetxt('./MSDS/test.csv', test_data, fmt='%.6f', delimiter=',')

# pruned and remedied SMD: SMD_partial does not need any change
# pruned and remedied MSL
values = pd.read_csv('./MSL/labeled_anomalies.csv')
values = values[values['spacecraft'] == 'MSL']
filenames = values['chan_id'].values.tolist()
test = np.load(f'./MSL/test/C-1.npy')
labels = np.zeros(test.shape[0])
indices = values[values['chan_id'] == 'C-1']['anomaly_sequences'].values[0]
indices = indices.replace(']', '').replace('[', '').split(', ')
indices = [int(i) for i in indices]
for i in range(0, len(indices), 2):
    labels[indices[i]:indices[i+1]] = 1
path = './MSL/labels/'
if not os.path.exists(path):
    os.makedirs(path)
print(labels.shape)
np.save(path + f'C-1.npy', labels)

# pruned and remedied SMAP
values = pd.read_csv('./SMAP/labeled_anomalies.csv')
values = values[values['spacecraft'] == 'SMAP']
filenames = values['chan_id'].values.tolist()
test = np.load(f'./SMAP/test/P-1.npy')
labels = np.zeros(test.shape[0])
indices = values[values['chan_id'] == 'P-1']['anomaly_sequences'].values[0]
indices = indices.replace(']', '').replace('[', '').split(', ')
indices = [int(i) for i in indices]
for i in range(0, len(indices), 2):
    labels[indices[i]:indices[i+1]] = 1
path = './SMAP/labels/'
if not os.path.exists(path):
    os.makedirs(path)
print(labels.shape)
np.save(path + f'P-1.npy', labels)

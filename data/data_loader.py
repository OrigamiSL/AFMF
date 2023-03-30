import os
import warnings

import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class Dataset_SMD(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/SMD', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = np.load(self.data_path + "/SMD_train.npy")
        data_test = np.load(self.data_path + "/SMD_test.npy")
        data_test_label = np.load(self.data_path + "/SMD_test_label.npy")

        if self.data_process:
            data_train = np.delete(data_train, 7, 1)
            data_test = np.delete(data_test, 7, 1)
        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)

        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        #
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes


class Dataset_MSL(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/MSL', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = np.load(self.data_path + "/MSL_train.npy")
        data_test = np.load(self.data_path + "/MSL_test.npy")
        data_test_label = np.load(self.data_path + "/MSL_test_label.npy")
        if self.data_process:
            data_train = np.concatenate([data_train[:, :1], data_train[:, [2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16,
                                                                           17, 19, 20, 23, 27, 28, 29, 31, 33, 35, 39,
                                                                           41, 42, 43, 45, 46, 47, 48, 49, 53, 54]]],
                                        axis=1)
            data_test = np.concatenate([data_test[:, :1], data_test[:, [2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16,
                                                                        17, 19, 20, 23, 27, 28, 29, 31, 33, 35, 39,
                                                                        41, 42, 43, 45, 46, 47, 48, 49, 53, 54]]],
                                       axis=1)
        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes


class Dataset_SMAP(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/SMAP', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = np.load(self.data_path + "/SMAP_train.npy")
        data_test = np.load(self.data_path + "/SMAP_test.npy")
        data_test_label = np.load(self.data_path + "/SMAP_test_label.npy")

        if self.data_process:
            data_train = np.delete(data_train, 16, 1)
            data_test = np.delete(data_test, 16, 1)
        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes


class Dataset_PSM(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/PSM', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = pd.read_csv(self.data_path + '/train.csv').values[:, 1:]
        data_test = pd.read_csv(self.data_path + '/test.csv').values[:, 1:]
        data_test_label = pd.read_csv(self.data_path + '/test_label.csv').values[:, 1:]

        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes


class Dataset_SWaT(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/SWaT', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = pd.read_csv(self.data_path + '/Normal.csv')
        data_test = pd.read_csv(self.data_path + '/Attack.csv')
        data_test_label = pd.read_csv(self.data_path + '/Attack.csv')

        if self.data_process:
            int_index = []
            for i in range(data_train.columns.shape[0]):
                int_index.append(str(int(float(data_train.columns[i]))))
            data_train.columns = int_index
            int_index.append(str(int(float(data_test.columns[-1]))))
            data_test.columns = int_index
            cols_dig = ['3', '4', '10', '13', '15', '20', '21', '22', '23', '24', '25', '31', '34',
                        '43', '50']
            cols_normal = ['1', '2', '6', '7', '8', '9', '17', '18', '19', '26', '27', '28', '29', '35', '36', '37',
                           '38',
                           '39', '40', '41', '42', '45', '46', '47', '48']
            data_train = data_train[cols_normal + cols_dig].values
            data_test = data_test[cols_normal + cols_dig].values
        else:
            data_train = data_train.values
            data_test = data_test.values[:, :-1]

        data_test_label = data_test_label.values[:, -1:]

        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        #
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes


class Dataset_WADI(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/WADI', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = pd.read_csv(self.data_path + '/Normal.csv')
        data_test = pd.read_csv(self.data_path + '/Attack.csv')
        data_test_label = pd.read_csv(self.data_path + '/Attack.csv')

        if self.data_process:
            int_index = []
            for i in range(data_train.columns.shape[0]):
                int_index.append(str(int(float(data_train.columns[i]))))
            data_train.columns = int_index
            int_index.append(str(int(float(data_test.columns[-1]))))
            data_test.columns = int_index
            cols_dig = ['10', '13', '14', '16', '18', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58',
                        '59', '71', '74', '76', '77', '78', '79', '80', '81', '83']
            cols_normal = ['1', '2', '3', '4', '5', '6', '9', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                           '29', '30', '31', '32',
                           '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
                           '60', '61', '63',
                           '64', '65', '66', '67', '68', '82', '84', '86', '87', '89', '90', '91', '98', '99', '100',
                           '101', '102',
                           '103', '104', '105', '107', '108', '109', '110', '111', '113', '121', '123']
            data_train = data_train[cols_normal + cols_dig].values
            data_test = data_test[cols_normal + cols_dig].values
        else:
            data_train = data_train.values
            data_test = data_test.values[:, :-1]

        data_test_label = data_test_label.values[:, -1:]

        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        #
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes


class Dataset_MBA(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/MBA', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = pd.read_excel(self.data_path + '/train.xlsx')
        data_test = pd.read_excel(self.data_path + '/test.xlsx')
        data_test_label = pd.read_excel(self.data_path + '/labels.xlsx')

        data_train = data_train.values[1:, 1:].astype(float)
        data_test = data_test.values[1:, 1:].astype(float)
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        data_test_label = data_test_label.values[:, 1].astype(int)

        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)

        labels = np.zeros_like(self.test)
        for i in range(-20, 20):
            labels[data_test_label + i, :] = 1
        self.test_labels = labels[:, :1]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        #
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes


class Dataset_UCR(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/UCR', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):
        file_name = '136_UCR_Anomaly_InternalBleeding17_1600_3198_3309.txt'
        data = pd.read_csv(self.data_path + '/' + file_name, header=None)
        vals = file_name.split('.')[0].split('_')
        vals = vals[-3:]
        self.vals = [int(i) for i in vals]
        data_train = data.values[:self.vals[0]]
        data_test = data.values[self.vals[0]:]

        data_test_label = np.zeros_like(data_test)
        data_test_label[self.vals[1] - self.vals[0]:self.vals[2] - self.vals[0]] = 1

        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]

        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test


class Dataset_NAB(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/NAB', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):
        data = pd.read_csv(self.data_path + '/ec2_request_latency_system_failure.csv')
        data_train = data[['value']].values
        data_test = data[['value']].values
        data_test_label = np.zeros_like(data_test)
        with open(self.data_path + '/labels.json') as f:
            labeldict = json.load(f)
        for timestamp in labeldict['realKnownCause/ec2_request_latency_system_failure.csv']:
            tstamp = timestamp.replace('.000000', '')
            index = np.where(((data['timestamp'] == tstamp).values + 0) == 1)[0][0]
            data_test_label[index - 4:index + 4] = 1

        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        #
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test


class Dataset_MSDS(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/MSDS', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = pd.read_csv(self.data_path + '/train.csv', header=None).values
        data_test = pd.read_csv(self.data_path + '/test.csv', header=None).values
        data_test_label = pd.read_csv(self.data_path + '/labels.csv', header=None).values

        data_train = np.nan_to_num(data_train)
        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test


class Dataset_SMD_partial(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/SMD', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = pd.read_csv(self.data_path + '/train/machine-1-1.txt', header=None).values
        data_test = pd.read_csv(self.data_path + '/test/machine-1-1.txt', header=None).values
        data_test_label = pd.read_csv(self.data_path + '/labels/machine-1-1.txt', header=None).values

        if self.data_process:
            data_train = np.delete(data_train, 7, 1)
            data_test = np.delete(data_test, 7, 1)
        data_train = np.nan_to_num(data_train)

        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)

        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        #
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes


class Dataset_MSL_partial(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/MSL', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = np.load(self.data_path + "/train/C-1.npy")
        data_test = np.load(self.data_path + "/test/C-1.npy")
        data_test_label = np.load(self.data_path + "/labels/C-1.npy")

        if self.data_process:
            data_train = np.concatenate([data_train[:, :1], data_train[:, [2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16,
                                                                           17, 19, 20, 23, 27, 28, 29, 31, 33, 35, 39,
                                                                           41, 42, 43, 45, 46, 47, 48, 49, 53, 54]]],
                                        axis=1)
            data_test = np.concatenate([data_test[:, :1], data_test[:, [2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16,
                                                                        17, 19, 20, 23, 27, 28, 29, 31, 33, 35, 39,
                                                                        41, 42, 43, 45, 46, 47, 48, 49, 53, 54]]],
                                       axis=1)
        data_train = np.nan_to_num(data_train)

        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes


class Dataset_SMAP_partial(Dataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/SMAP', data_process=False, LIN=True,
                 partial_train=False, ratio=0.2):
        # info
        self.input_len = input_len
        self.LIN = LIN
        if not LIN:
            self.scaler = StandardScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1

        self.__read_data__()

    def __read_data__(self):

        data_train = np.load(self.data_path + "/train/P-1.npy")
        data_test = np.load(self.data_path + "/test/P-1.npy")
        data_test_label = np.load(self.data_path + "/labels/P-1.npy")

        if self.data_process:
            data_train = np.delete(data_train, 16, 1)
            data_test = np.delete(data_test, 16, 1)
        data_train = np.nan_to_num(data_train)

        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        if not self.LIN:
            self.scaler.fit(data_train)
            data_train = self.scaler.transform(data_train)
            data_train = np.nan_to_num(data_train)
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]

        self.test = np.nan_to_num(data_test)
        if not self.LIN:
            self.test = self.scaler.transform(self.test)
            self.test = np.nan_to_num(self.test)
        self.test_labels = data_test_label

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        seq_x = np.nan_to_num(seq_x)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test

    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    # index_feature_map.append(child)

                c_index = index_feature_map.index(child)
                # edge_indexes[0].append(p_index)
                # edge_indexes[1].append(c_index)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes

from data.data_loader import Dataset_SMD, Dataset_MSL, Dataset_SMAP, Dataset_PSM, Dataset_SWaT, Dataset_WADI, \
    Dataset_MBA
from exp.exp_basic import Exp_Basic
from models.RT.model import RF
from models.DLinear.DLinear import DLinear
from models.Autoformer.Autoformer import Autoformer
from models.DeepAR.model import DeepAR
from models.GTA.gta import GTA
from models.Informer.model import Informer
from models.LSTNet.model import LSTNet
from models.MA.MA import MA
from models.MTAD_GAT.MTAD_GAT import MTAD_GAT
from models.GDN.GDN import GDN

from utils.tools import EarlyStopping, adjust_learning_rate, \
    loss_process, detection_adjustment, anomaly_adjustment

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, auc

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self, device):
        model_dict = {
            'RTNet': RF,
            'DLinear': DLinear,
            'Autoformer': Autoformer,
            'DeepAR': DeepAR,
            'GTA': GTA,
            'Informer': Informer,
            'LSTNet': LSTNet,
            'MA': MA,
            'MTAD_GAT': MTAD_GAT,
            'GDN': GDN
        }
        if self.args.model == 'RTNet':
            model = model_dict[self.args.model](
                self.args.variate,
                self.args.out_variate,
                self.args.input_len,
                self.args.kernel,
                self.args.block_nums,
                self.args.d_model,
                self.args.pyramid,
                self.args.LIN,
                self.args.dropout
            ).float()
        elif self.args.model == 'DLinear':
            model = model_dict[self.args.model](
                self.args.variate,
                self.args.out_variate,
                self.args.input_len,
                self.args.kernel,
                self.args.LIN
            ).float()
        elif self.args.model == 'Autoformer':
            model = model_dict[self.args.model](
                self.args.variate,
                self.args.out_variate,
                self.args.input_len,
                self.args.label_len,
                self.args.moving_avg,
                self.args.d_model,
                self.args.dropout,
                self.args.factor,
                self.args.n_heads,
                self.args.activation,
                self.args.e_layers,
                self.args.d_layers,
                self.args.LIN
            ).float()
        elif self.args.model == 'DeepAR':
            model = model_dict[self.args.model](
                self.args.variate,
                self.args.out_variate,
                self.args.input_len,
                self.args.d_model,
                self.args.num_layers,
                self.args.LIN
            ).float()
        elif self.args.model == 'GTA':
            model = model_dict[self.args.model](
                self.args.variate,
                self.args.out_variate,
                self.args.input_len,
                self.args.label_len,
                self.args.num_levels,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.dropout,
                self.args.activation,
                self.args.LIN,
                device
            ).float()
        elif self.args.model == 'Informer':
            model = model_dict[self.args.model](
                self.args.variate,
                self.args.variate,
                self.args.out_variate,
                self.args.input_len,
                self.args.label_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.activation,
                self.args.distil,
                self.args.mix,
                self.args.LIN
            ).float()
        elif self.args.model == 'LSTNet':
            model = model_dict[self.args.model](
                self.args.input_len,
                self.args.variate,
                self.args.RNN_hid_size,
                self.args.CNN_hid_size,
                self.args.hidSkip,
                self.args.skip,
                self.args.CNN_kernel,
                self.args.highway_window,
                self.args.dropout,
                self.args.out_variate,
                self.args.LIN
            ).float()
        elif self.args.model == 'MA':
            model = model_dict[self.args.model](
                self.args.out_variate,
                self.args.input_len,
                self.args.LIN,
            ).float()
        elif self.args.model == 'MTAD_GAT':
            model = model_dict[self.args.model](
                self.args.variate,
                self.args.out_variate,
                self.args.input_len,
                self.args.kernel,
                self.args.embed_dim,
                self.args.time_embed_dim,
                self.args.use_gatv2,
                self.args.gru_n_layers,
                self.args.gru_hid_dim,
                self.args.fc_n_layers,
                self.args.fc_hid_dim,
                self.args.recon_n_layers,
                self.args.recon_hid_dim,
                self.args.dropout,
                self.args.alpha,
                self.args.LIN,
            ).float()
        elif self.args.model == 'GDN':
            fc_edge_index = self._get_data(flag='train', graph=True)
            fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
            self.edge_index_sets = []
            self.edge_index_sets.append(fc_edge_index)
            model = model_dict[self.args.model](
                self.args.variate,
                self.args.out_variate,
                self.args.input_len,
                self.edge_index_sets,
                self.args.d_model,
                self.args.out_layer_inter_dim,
                self.args.out_layer_num,
                self.args.topk,
                self.args.dropout,
                self.args.LIN
            ).float()
        return model.to(device)

    def _get_data(self, flag, graph=False):
        args = self.args

        data_dict = {
            'SMD': Dataset_SMD,
            'MSL': Dataset_MSL,
            'SMAP': Dataset_SMAP,
            'PSM': Dataset_PSM,
            'SWaT': Dataset_SWaT,
            'WADI': Dataset_WADI,
            'MBA': Dataset_MBA,
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
        else:
            shuffle_flag = True
            drop_last = True

        data_set = Data(
            flag=flag,
            input_len=args.input_len,
            data_path=args.data_path,
            data_process=args.data_process,
            LIN=args.LIN,
            partial_train=args.partial_train,
            ratio=args.partial_ratio,
        )
        print(flag, len(data_set))
        num_workers = 0
        if graph:
            edge_index_sets = data_set.build_loc_net()
            return edge_index_sets
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=num_workers,
                drop_last=drop_last)

            return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.AdamW(self.model.parameters(), weight_decay=1e-5, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data=None, vali_loader=None, criterion=None):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x) in enumerate(vali_loader):
                if self.args.model == 'MTAD_GAT':
                    pred, recon, true = self._process_one_batch(batch_x)
                    pred_loss = loss_process(pred, true, criterion, flag=1)
                    recon_loss = loss_process(recon, true, criterion, flag=1)
                    loss = pred_loss + recon_loss
                else:
                    pred, true = self._process_one_batch(batch_x)
                    loss = loss_process(pred, true, criterion, flag=1)
                total_loss.append(loss)

            total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        if self.args.data_process:
            path = os.path.join(self.args.checkpoints, 'process', self.args.model, self.args.data, setting)
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = os.path.join(self.args.checkpoints, 'not_process', self.args.model, self.args.data, setting)
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if not self.args.load_anomaly:
            print('-' * 99)
            print('starting training')
            lr = self.args.learning_rate
            for epoch in range(self.args.train_epochs):
                iter_count = 0

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x) in enumerate(train_loader):
                    model_optim.zero_grad()
                    iter_count += 1
                    if self.args.model == 'MTAD_GAT':
                        pred, recon, true = self._process_one_batch(
                            batch_x)
                        pred_loss = loss_process(pred, true, criterion, flag=0)
                        recon_loss = loss_process(recon, true, criterion, flag=0)
                        loss = pred_loss + recon_loss
                    else:
                        pred, true = self._process_one_batch(
                            batch_x)
                        loss = loss_process(pred, true, criterion, flag=0)

                    loss.backward(torch.ones_like(loss))
                    model_optim.step()

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                                torch.mean(loss).item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                vali_loss = self.vali(vali_data, vali_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Vali Loss: {2:.7f}".format(
                    epoch + 1, train_steps, vali_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

            self.args.learning_rate = lr
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, load=True, load_anomaly=False):
        self.model.eval()
        if load:
            if self.args.data_process:
                path = os.path.join(self.args.checkpoints, 'process', self.args.model, self.args.data, setting)
            else:
                path = os.path.join(self.args.checkpoints, 'not_process', self.args.model, self.args.data, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        if self.args.data_process:
            folder_path = './results/process/' + self.args.model + '/' + self.args.data + '/'
            comparison_path = './comparison/process/' + self.args.model + '/' + self.args.data + '/'
        else:
            folder_path = './results/not_process/' + self.args.model + '/' + self.args.data + '/'
            comparison_path = './comparison/not_process/' + self.args.model + '/' + self.args.data + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(comparison_path):
            os.makedirs(comparison_path)

        test_data, test_loader = self._get_data(flag='test')
        test_label = test_data.get_label()

        mse = []
        preds = []
        trues = []
        recons = []
        if self.args.drop:
            mse_drop_all = dict.fromkeys(range(self.args.drop), [])
            for ind in mse_drop_all:
                mse_drop_all[ind] = []
            preds_drop = dict.fromkeys(range(self.args.drop), [])
            for ind in preds_drop:
                preds_drop[ind] = []
            recons_drop = dict.fromkeys(range(self.args.drop), [])
            for ind in recons_drop:
                recons_drop[ind] = []

        print('test steps for forecasting:', len(test_loader))
        if not load_anomaly:
            with torch.no_grad():
                print('Starting testing')
                time_test = time.time()
                for i, (batch_x) in enumerate(test_loader):
                    if self.args.model == 'MTAD_GAT':
                        pred, recon, true = self._process_one_batch(batch_x)
                        recon = recon.squeeze().detach().cpu().numpy()
                        pred = pred.squeeze().detach().cpu().numpy()
                        true = true.squeeze().detach().cpu().numpy()
                        t_mse = (pred - true) ** 2 + (recon - true) ** 2
                    else:
                        pred, true = self._process_one_batch(batch_x)
                        pred = pred.squeeze().detach().cpu().numpy()
                        true = true.squeeze().detach().cpu().numpy()
                        t_mse = (pred - true) ** 2
                    mse.append(t_mse)
                    if self.args.drop:
                        for ind in mse_drop_all:
                            if self.args.model == 'MTAD_GAT':
                                pred_drop, recon_drop, _ = self._process_one_batch(batch_x, drop=ind + 1)
                                pred_drop = pred_drop.squeeze().detach().cpu().numpy()
                                recon_drop = recon_drop.squeeze().detach().cpu().numpy()
                                t_mse_drop = (pred_drop - true) ** 2 + (recon_drop - true) ** 2
                            else:
                                pred_drop, _ = self._process_one_batch(batch_x, drop=ind + 1)
                                pred_drop = pred_drop.squeeze().detach().cpu().numpy()
                                t_mse_drop = (pred_drop - true) ** 2
                            mse_drop_all[ind].append(t_mse_drop)
                            if self.args.save_predictions:
                                preds_drop[ind].append(pred_drop)
                                recons_drop[ind].append(recon_drop)
                    if self.args.save_predictions:
                        preds.append(pred)
                        trues.append(true)
                        if self.args.model == 'MTAD_GAT':
                            recons.append(recon)
                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_test) / 100
                        print("\titers: {}| speed: {:.4f}s/iter".format(i + 1, speed))
                        time_test = time.time()

        else:
            mse = np.load(folder_path + "/mse.npy")
            if self.args.drop:
                for key in range(self.args.drop):
                    mse_drop_all[key] = np.load(folder_path + f"/mse_drop{int(key)}.npy")

        mse = np.array(mse)
        print('test shape:', mse.shape)
        print('mse_loss:', np.mean(mse))
        mse = mse.reshape(-1, self.args.out_variate)

        print('test shape:', mse.shape)

        if not load_anomaly:
            if self.args.save_mses:
                np.save(folder_path + f'mse.npy', mse)
                np.savetxt(comparison_path + f'mse.csv', mse, fmt='%.6f', delimiter=',')
            if self.args.drop:
                for key in range(self.args.drop):
                    if self.args.save_mses:
                        np.save(folder_path + f'mse_drop{int(key)}.npy', mse_drop_all[key])

            if self.args.save_predictions:
                preds = np.array(preds)
                preds = preds.reshape(-1, self.args.out_variate)
                np.save(folder_path + f'pred.npy', preds)

                trues = np.array(trues)
                trues = trues.reshape(-1, self.args.out_variate)
                np.save(folder_path + f'true.npy', trues)

                if self.args.model == 'MTAD_GAT':
                    recons = np.array(recons)
                    recons = recons.reshape(-1, self.args.out_variate)
                    np.save(folder_path + f'recon.npy', recons)

                if self.args.drop:
                    for key in range(self.args.drop):
                        cur_preds_drop = np.array(preds_drop[key])
                        cur_preds_drop = cur_preds_drop.reshape(-1, self.args.out_variate)
                        np.save(folder_path + f"pred_drop{int(key)}.npy", cur_preds_drop)
                        np.savetxt(comparison_path + f'pred_drop{int(key)}.csv', cur_preds_drop,
                                   fmt='%.6f', delimiter=',')

                np.savetxt(comparison_path + f'pred_ini.csv', preds, fmt='%.6f', delimiter=',')
                np.savetxt(comparison_path + f'true.csv', trues, fmt='%.6f', delimiter=',')

        mse_init = np.mean(mse, -1)
        mse_all = mse.copy()
        mse_thresh = np.percentile(mse_init, 100 - self.args.anomaly_ratio)
        anomaly = (mse_init > mse_thresh).astype(int)
        np.savetxt(comparison_path + f'anomaly_ini.csv', anomaly, fmt='%.6f', delimiter=',')
        print('forecasting thres: ', mse_thresh)
        if self.args.drop:
            mse, k_list = anomaly_adjustment(anomaly, mse_drop_all, mse, self.args.out_variate, self.args.thresh,
                                             self.args.drop)
            k_list = k_list.reshape(-1)
            no_k = np.where(k_list == -1)[0].shape[0]
            zero_k = np.where(k_list == 0)[0].shape[0]
            one_k = np.where(k_list == 1)[0].shape[0]
            two_k = np.where(k_list == 2)[0].shape[0]
            three_k = np.where(k_list == 3)[0].shape[0]
            four_k = np.where(k_list == 4)[0].shape[0]
            over_four_k = np.where(k_list > 4)[0].shape[0]
            print("distributions of final k values: -1_{}| 0_{} | 1_{}| 2_{} | 3_{} | 4_{} | >4_{}".format(
                no_k, zero_k, one_k, two_k, three_k, four_k, over_four_k
            ))

            if self.args.save_mses:
                np.savetxt(comparison_path + f'mse_final.csv', mse, fmt='%.6f', delimiter=',')
            mse = np.mean(mse, -1)
            anomaly = (mse > mse_thresh).astype(int)
            np.savetxt(comparison_path + f'anomaly_final.csv', anomaly, fmt='%.6f', delimiter=',')
        former_len = self.args.input_len - 1
        gt = test_label[former_len:former_len + anomaly.shape[0]].astype(int)
        print('test_anomaly shape:', anomaly.shape, gt.shape)
        if self.args.detection_adjustment:
            anomaly, gt = detection_adjustment(anomaly, gt)

        if self.args.drop:
            roc_auc, tps, fps = self._roc_auc_all(gt, mse_all, mse_drop_all)
        else:
            roc_auc, tps, fps = self._roc_auc_all(gt, mse_all)

        np.save(folder_path + f'anomaly.npy', anomaly)
        np.save(folder_path + f'gt.npy', gt)
        np.savetxt(comparison_path + f'anomaly_pa.csv', anomaly, fmt='%.6f', delimiter=',')
        np.savetxt(comparison_path + f'gt.csv', gt, fmt='%.6f', delimiter=',')

        accuracy = accuracy_score(gt, anomaly)
        precision, recall, f_score, support = \
            precision_recall_fscore_support(gt, anomaly, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score, roc_auc))
        path = './result.log'
        with open(path, "a") as f:
            f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
            f.write('|model_{}|data_{}|data_preprocess_{}|drop_num{}| P:,{}, R:,{}, F1:,{}, AUC:,{}'.
                    format(self.args.model, self.args.data, self.args.data_process, self.args.drop,
                           precision, recall, f_score, roc_auc) + '\n')
            f.write('|tp:,{}, fp:,{}'.
                    format(tps, fps) + '\n')
            f.flush()
            f.close()
        return precision, recall, f_score, roc_auc

    def _process_one_batch(self, batch_x, drop=0):
        batch_x = batch_x.float().to(self.device)
        if self.args.model == 'MTAD_GAT':
            outputs, recon, gt = self.model(batch_x, drop)
            return outputs, recon, gt
        else:
            outputs, gt = self.model(batch_x, drop)
            return outputs, gt

    def _roc_auc_all(self, gt, mse_all, mse_drop_all=None):
        thresholds = [0.5, 1.0, 1.5, 2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        n_pos = gt.sum()
        n_neg = (1 - gt).sum()
        tps = []
        fps = []
        for thre in thresholds:
            mse_init = np.mean(mse_all, -1)
            mse_thresh = np.percentile(mse_init, 100 - thre)
            anomaly = (mse_init > mse_thresh).astype(int)
            if self.args.drop:
                mse, _ = anomaly_adjustment(anomaly, mse_drop_all, mse_all, self.args.out_variate, self.args.thresh,
                                         self.args.drop)
                mse = np.mean(mse, -1)
                anomaly = (mse > mse_thresh).astype(int)

            if self.args.detection_adjustment:
                anomaly, gt = detection_adjustment(anomaly, gt)

            pred_pos = np.argwhere(anomaly == 1)

            tp = gt[pred_pos].sum() / n_pos
            fp = (1 - gt[pred_pos]).sum() / n_neg

            tps.append(tp)
            fps.append(fp)

        tps.append(1.0)
        fps.append(1.0)

        auc_score = auc(fps, tps)
        return auc_score, tps, fps

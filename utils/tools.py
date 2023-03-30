import numpy as np
import torch
import torch.nn.functional as F


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        args.learning_rate = args.learning_rate * 0.5
        lr_adjust = {epoch: args.learning_rate}

    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def loss_process(pred, true, criterion=None, flag=0):
    if flag == 0:
        loss = criterion(pred, true)
        return loss
    else:
        loss2 = criterion(pred, true)
        return loss2.detach().cpu().numpy()


def detection_adjustment(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    return pred, gt


def pak(scores, targets, k=20):
    one_start_idx = np.where(np.diff(targets, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(targets, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(scores))

    for i in range(len(one_start_idx)):
        if scores[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            scores[one_start_idx[i]:zero_start_idx[i]] = 1

    return scores, targets


def anomaly_adjustment(anomaly, mse_drop_all, mse, variate, thresh=3, drop=1):
    mse_decrease = dict.fromkeys(range(drop), [])
    for key in mse_drop_all:
        mse_drop_all[key] = np.array(mse_drop_all[key])
        mse_drop_all[key] = mse_drop_all[key].reshape(-1, variate)
        mse_decrease[key] = mse / mse_drop_all[key]
    k_value = []
    for ind in range(variate):
        for i in range(len(anomaly)):
            if anomaly[i] == 1:
                min_d = -1
                max_decrease = thresh
                for d in range(drop):
                    if mse_decrease[d][i, ind] > max_decrease:
                        mse[i, ind] = mse_drop_all[d][i, ind]
                        max_decrease = mse_decrease[d][i, ind]
                        min_d = d
                    else:
                        continue
                k_value.append(min_d)
    return mse, np.array(k_value)

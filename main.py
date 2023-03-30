import argparse
import os
import torch
import numpy as np
import time

from exp.exp_model import Exp_Model

parser = argparse.ArgumentParser(description='[AFMF]')
# model selection
parser.add_argument('--model', type=str, required=True, default='RTNet',
                    help='model of experiment')

# data related
parser.add_argument('--data', type=str, required=True, default='SMD', help='data')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--data_path', type=str, default='./data/SMD', help='root path of the data file')

# common setting
parser.add_argument('--input_len', type=int, default=720, help='input sequence length of the model')
parser.add_argument('--label_len', type=int, default=48, help='input sequence length of the model')
parser.add_argument('--variate', type=int, default=38, help='input variate number')
parser.add_argument('--out_variate', type=int, default=37, help='output variate number')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--anomaly_ratio', type=float, default=1.5, help='anomaly_ratio')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

parser.add_argument('--detection_adjustment', action='store_true',
                    help='whether to use detection_adjustment'
                    , default=False)
parser.add_argument('--partial_train', action='store_true',
                    help='whether to use partial train subset',
                    default=False)
parser.add_argument('--partial_ratio', type=float, default=0.2)
parser.add_argument('--partial_data', action='store_true',
                    help='whether to only use partial data which are not flawed for experiment',
                    default=False)
parser.add_argument('--adjust_k', type=int, default=0)

# AFMF
parser.add_argument('--LIN', action='store_true',
                    help='whether to use local instance normalization'
                    , default=False)
parser.add_argument('--data_process', action='store_true',
                    help='whether to preprocess data'
                    , default=False)
parser.add_argument('--drop', type=int, default=0, help='loop variate k')
parser.add_argument('--thresh', type=float, default=3, help='decline ratio')

# RTNet
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--pyramid', type=int, default=1)
parser.add_argument('--kernel', type=int, default=3)
parser.add_argument('--block_nums', type=int, default=3)

# Autoformer (shared by GTA and Informer)
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--e_layers', type=int, default=2, help='encoder layer')
parser.add_argument('--d_layers', type=int, default=1, help='decoder layer')
parser.add_argument('--n_heads', type=int, default=8, help='')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of model')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--factor', type=int, default=1, help='attn factor')

# DeepAR
parser.add_argument('--num_layers', type=int, default=3)

# GTA
parser.add_argument('--num_levels', type=int, default=3, help='number of dilated levels for graph embedding')

# Informer
parser.add_argument('--attn', type=str, default='prob', help='activation')
parser.add_argument('--distil', action='store_true',
                    help='whether to use distil operation',
                    default=False)
parser.add_argument('--mix', action='store_true',
                    help='whether to mix after attention',
                    default=False)

# LSTNet
parser.add_argument('--RNN_hid_size', default=512, type=int, help='hidden channel of RNN module')
parser.add_argument('--CNN_hid_size', type=int, default=100, help='number of CNN hidden units')
parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=0, help='The window size of the highway component')
parser.add_argument('--skip', type=int, default=24)
parser.add_argument('--hidSkip', type=int, default=5)

# MTAD-GAT
parser.add_argument("--use_gatv2", action='store_true', help='', default=False)
parser.add_argument("--embed_dim", type=int, default=None)
parser.add_argument("--time_embed_dim", type=int, default=None)
parser.add_argument("--gru_n_layers", type=int, default=1)
parser.add_argument("--gru_hid_dim", type=int, default=150)
parser.add_argument("--fc_n_layers", type=int, default=3)
parser.add_argument("--fc_hid_dim", type=int, default=150)
parser.add_argument("--recon_n_layers", type=int, default=1)
parser.add_argument("--recon_hid_dim", type=int, default=150)
parser.add_argument("--alpha", type=float, default=0.2)

# GDN
parser.add_argument('--out_layer_num', help='outlayer num', type = int, default=1)
parser.add_argument('--out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
parser.add_argument('--topk', help='topk num', type = int, default=20)

# Experimental setting
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--retrain', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--load_anomaly', action='store_true',
                    help='whether to load anomaly'
                    , default=False)
parser.add_argument('--save_predictions', action='store_true',
                    help='whether to save prediction results',
                    default=False)
parser.add_argument('--save_mses', action='store_true',
                    help='whether to save mses',
                    default=False)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)


args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

lr = args.learning_rate
print('Args in experiment:')
print(args)

P_list = []
R_list = []
F_list = []
AUC_list = []
for ii in range(args.itr):
    # setting record of experiments

    setting = '{}_{}_il{}_{}'.format(args.model,
                                     args.data,
                                     args.input_len,
                                     ii)
    Exp = Exp_Model
    exp = Exp(args)  # set experiments
    if args.retrain and args.model != 'MA':
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        try:
            exp.train(setting)
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    if args.model != 'MA':
        P, R, F, AUC = exp.test(setting, load=True, load_anomaly=args.load_anomaly)
    else:
        P, R, F, AUC = exp.test(setting, load=False, load_anomaly=args.load_anomaly)
    P_list.append(P)
    R_list.append(R)
    F_list.append(F)
    AUC_list.append(AUC)

    torch.cuda.empty_cache()
    args.learning_rate = lr

P_list = np.asarray(P_list)
R_list = np.asarray(R_list)
F_list = np.asarray(F_list)
AUC_list = np.asarray(AUC_list)
avg_P = np.mean(P_list)
std_P = np.std(P_list)
avg_R = np.mean(R_list)
std_R = np.std(R_list)
avg_F = np.mean(F_list)
std_F = np.std(F_list)
avg_AUC = np.mean(AUC_list)
std_AUC = np.std(AUC_list)
path = './result.log'
with open(path, "a") as f:
    f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    f.write('|model_{}|data_{}|data_preprocess_{}|drop_num{}|'.
            format(args.model, args.data, args.data_process, args.drop) + '\n')
    f.write('P|mean_{}|std_{}'.format(avg_P, std_P) + '\n')
    f.write('R|mean_{}|std_{}'.format(avg_R, std_R) + '\n')
    f.write('F|mean_{}|std_{}'.format(avg_F, std_F) + '\n')
    f.write('AUC|mean_{}|std_{}'.format(avg_AUC, std_AUC) + '\n')
    f.flush()
    f.close()
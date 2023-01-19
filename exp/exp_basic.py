import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            self.args.gpu)
        device = torch.device('cuda:{}'.format(self.args.gpu))
        print('Use GPU: cuda:{}'.format(self.args.gpu))
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

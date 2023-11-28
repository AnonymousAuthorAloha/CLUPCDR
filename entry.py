import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"
import time
import torch
import numpy as np
import random
import argparse
import json
# from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from my_run import Run
import sys
from datetime import datetime


def prepare(config_path):
# 很常规的操作 是一个定式 拿来即用即可 不用特别去记忆
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--ratio', default=[0.8, 0.2])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=2)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('-data', metavar='DIR', default='./datasets',help='path to dataset')
    # parser.add_argument('-dataset-name', default='stl10', help='dataset name', choices=['stl10', 'cifar10'])
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',choices=model_names,help='model architecture: ' +
    #                         ' | '.join(model_names) +
    #                         ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    # parser.add_argument('--epochs', default=200, type=int, metavar='N',
    #                     help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # parser.add_argument('--seed', default=None, type=int,
    #                     help='seed for initializing training. ')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')

    parser.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    args = parser.parse_args()

# 设置种子常规操作
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(config_path, 'r') as f:
        config = json.load(f)
        config['base_model'] = args.base_model
        config['task'] = args.task
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = 0.01
    return args, config

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

if __name__ == '__main__':
    config_path = 'config.json'
    args, config = prepare(config_path)
# 在 Python 的 os 模块中，os.environ 是一个环境变量的映射对象，它允许我们获取和设置环境变量。
# os.environ['CUDA_VISIBLE_DEVICES'] 是一个环境变量，用于指定在进行 GPU 计算时，哪些 GPU 设备是可见的，即可以被使用的。
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu   
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    current_date = datetime.now().date()

    # date_string = current_date.strftime("%Y-%m-%d")
    date_string ="all_experiment_t_002_nolinearEMCDR_augmentation"
    path=r'.\%s.txt'%date_string
    path_result=r'.\%s_result.txt'%date_string
    sys.stdout = Logger(path)
    for config['task'] in ['1','2','3']:
        for config['ratio'] in  [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
        # for config['ratio'] in  [[0.8, 0.2]]:
            print("task:"+config["src_tgt_pairs"][config['task']]['src']+" and "+config["src_tgt_pairs"][config['task']]["tgt"])
            print("Ration:"+str(config['ratio']))
            print("lr:"+str(config['lr']))
            config['epoch']=50
            print('epoch：'+str(config['epoch']))
            print("temperature:",0.02)
            run = Run(config,0.02)
            run.main()

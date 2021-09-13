# @Author:Yifx
# @Contact: Xxuyifan1999@163.com
# @Time:2021/9/11 16:01
# @Software: PyCharm

"""
文件说明：
"""
import time
import torch
import numpy as np
from src.train_eval import train
from importlib import import_module
import argparse
from config.config import Config
from utils import build_vocab,build_dataset,DataIterator


# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# args = parser.parse_args()

config = Config()
import torch.nn as nn
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    model_name = "TextRNN"
    model = import_module('models.' + model_name)

    print("Loading data...")
    vocab,train_data,dev_data,test_data = build_dataset(config)
    train_iter = DataIterator(train_data,config.batch_size)
    dev_iter = DataIterator(dev_data,config.batch_size)
    test_iter = DataIterator(test_data,config.batch_size)

    config.n_vocab = len(vocab)

    model = model.TextRNN(config).to(config.device)
    init_network(model)
    train(config,model,train_iter,dev_iter,test_iter)







# @Author:Yifx
# @Contact: Xxuyifan1999@163.com
# @Time:2021/9/11 14:35
# @Software: PyCharm

"""
文件说明：
"""
import torch
import numpy as np

# 参数整理
class Config(object):
    def __init__(self):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.MAX_VOCAB_SIZE = 50000 # 词表最大容量
        self.MIN_FREQ = 1
        self.PAD_SIZE = 32
        self.num_classes = 10

        self.train_path = '../data/train.txt'
        self.dev_path = '../data/dev.txt'
        self.test_path = '../data/test.txt'
        self.vocab_path = '../data/vocab.pkl'
        self.model_path = '../models/best_model'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 64

        self.embedding_path = "../data/embedding_SougouNews.npz"

        self.embedding_pretrained = torch.tensor(np.load(self.embedding_path)["embeddings"].astype('float32'))
        self.embed_dim = self.embedding_pretrained.size()[1]
        self.hidden_dim = 100
        self.drop_out = 0
        self.n_vocab = 0 # 运行时赋值

        self.nums_epochs = 1000
        self.learning_rate = 0.0001
        self.require_improvement = 10000

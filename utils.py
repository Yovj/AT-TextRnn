# @Author:Yifx
# @Contact: Xxuyifan1999@163.com
# @Time:2021/9/11 14:27
# @Software: PyCharm

"""
文件说明：一些工具函数
"""
import torch
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
from config.config import Config

import time
from datetime import timedelta

config = Config()

def build_vocab(file_path,tokenizer,max_vocab_size,min_freq):
    # 生成词表

    vocab_dic = {}
    with open(file_path,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            text = lin.split('\t')[0]
            # 统计词频
            for word in tokenizer(text):
                vocab_dic[word] = vocab_dic.get(word,0) + 1

        vocab_list = sorted([x for x in vocab_dic.items() if x[1] >= min_freq],key = lambda x:x[1])[:max_vocab_size]
        vocab_dic = {word:idx for idx,(word,_) in enumerate(vocab_list)}
        vocab_dic.update({config.PAD:len(vocab_list),
                          config.UNK:len(vocab_list)+1})

    return vocab_dic



def build_dataset(config,pad_size=32):
    # 生成 Dataset
    tokenizer = lambda x:[y for y in x]
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path,'rb'))
    else:
        vocab = build_vocab(config.train_path,tokenizer,config.MAX_VOCAB_SIZE,config.MIN_FREQ)
        pkl.dump(vocab,open(config.vocab_path,'wb'))

    def load_dataset(file_path):
        contents = []

        with open(file_path,'r',encoding='utf-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                text,label = lin.split('\t')
                content = tokenizer(text)
                seq_len = len(text)
                # str to id
                token = [vocab.get(word,vocab.get(config.UNK)) for word in content]
                # padding
                if seq_len < pad_size:
                    token.extend([vocab.get(config.UNK) for i in range(pad_size - seq_len)])
                else:
                    token = token[:pad_size]

                contents.append((token,seq_len,int(label)))
        return contents

    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return vocab,train,dev,test





class DataIterator(object):
    def __init__(self,batches,batch_size):
        self.batches = batches
        self.n_batch = len(batches) // batch_size
        self.batch_size = batch_size
        self.residue = True if len(batches)% batch_size == 0 else False
        self.index = 0 # 指示当前 batch 的data索引

    def __len__(self):
        return self.n_batch if self.residue == 0 else self.n_batch+1

    # 采用生成器的方法产生数据
    # def __iter__(self):
    #     for self.index in range(0,self.n_batch + 1):
    #         if self.index == self.n_batch and self.residue:
    #             batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size]
    #             batches = self.to_tensor(batches)
    #             self.index += 1
    #             yield batches
    #             break
    #         batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size]
    #         batches = self.to_tensor(batches)
    #         self.index += 1
    #         yield batches
    #     self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.n_batch and self.residue:
            batches = self.batches[self.index * self.batch_size:]
            self.index += 1
            batches = self.to_tensor(batches)
            return batches
        elif self.index >= self.n_batch:
            # 数据迭代完成
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            batches = self.to_tensor(batches)
            self.index += 1
            return batches

    def to_tensor(self,batches):
        x = torch.LongTensor([x[0] for x in batches]).to(config.device)
        y = torch.LongTensor([x[2] for x in batches]).to(config.device)
        seq_len = torch.LongTensor([x[1] for x in batches]).to(config.device)
        return (x,seq_len),y



def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds = int(round(time_dif)))


if __name__ == '__main__':
    tokenizer = lambda x:[y for y in x]
    # vocab = build_vocab(config.train_path,tokenizer,config.MAX_VOCAB_SIZE,config.MIN_FREQ)
    # print(vocab)
    vocab,train,dev,test = build_dataset(config)
    train_iter = DataIterator(train,config.batch_size)
    print((next(train_iter)))
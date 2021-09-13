# @Author:Yifx
# @Contact: Xxuyifan1999@163.com
# @Time:2021/9/11 16:01
# @Software: PyCharm

"""
文件说明：
"""
import torch
import torch.nn as nn
import numpy as np
from config.config import Config

# No - Attention - Version
# class TextRNN(nn.Module):
#     def __init__(self,config):
#         super(TextRNN,self).__init__()
#         if config.embedding_path:
#             print("aaaa")
#             self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.MAX_VOCAB_SIZE,config.embed_dim,padding_idx=config.MAX_VOCAB_SIZE-2)
#
#         # 注意，如果使用预训练的词向量， config.embed_dim 需要等于预训练词向量维度
#         self.lstm = nn.LSTM(config.embed_dim,config.hidden_dim,batch_first=True,bidirectional=True,dropout=config.drop_out)
#         self.fc = nn.Linear(2*config.hidden_dim,config.num_classes)
#         self.softmax = nn.Softmax(dim=-1)
#
#
#     def forward(self,batches):
#         x = batches
#         out = self.embedding(x)
#         out,(hidden,cell) = self.lstm(out)
#
#         out = self.fc(out[:,-1,:])
#         out = self.softmax(out)
#
#         return out



class TextRNN(nn.Module):
    def __init__(self,config):
        super(TextRNN,self).__init__()
        self.config = config
        if config.embedding_path:
            print("aaaa")
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding = nn.Embedding(config.MAX_VOCAB_SIZE,config.embed_dim,padding_idx=config.MAX_VOCAB_SIZE-2)

        # 注意，如果使用预训练的词向量， config.embed_dim 需要等于预训练词向量维度
        self.lstm = nn.LSTM(config.embed_dim,config.hidden_dim,batch_first=True,bidirectional=True,dropout=config.drop_out)
        self.fc = nn.Linear(2*config.hidden_dim,config.num_classes)
        self.softmax = nn.Softmax(dim=-1)

        #######
        self.w_omiga = torch.randn(config.batch_size,2*config.hidden_dim,1,requires_grad=True).to(config.device)

    def forward(self,batches):
        x = batches
        out = self.embedding(x)
        out,(hidden,cell) = self.lstm(out)

        H = torch.nn.Tanh()(out)
        weights = torch.nn.Softmax(dim=-1)(torch.bmm(H,self.w_omiga).squeeze()).unsqueeze(dim=-1).repeat(1,1,self.config.hidden_dim * 2)
        out = torch.mul(out,weights).sum(dim=-2)

        out = self.fc(out)
        out = self.softmax(out)

        return out

if __name__ == '__main__':
    config = Config()
    text_rnn = TextRNN(config)
    x = torch.rand(8,32).long()
    y = text_rnn(x)
    print(y.shape)
    print(y)
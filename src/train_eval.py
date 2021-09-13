# @Author:Yifx
# @Contact: Xxuyifan1999@163.com
# @Time:2021/9/11 16:01
# @Software: PyCharm

"""
文件说明：
"""
from config.config import Config
import torch
import torch.nn.functional as F
import time
from sklearn import metrics
from utils import get_time_dif
import numpy as np



def train(config,model,train_iter,dev_iter,test_iter):
    start_time = time.time()
    model.train()

    total_batch = 0
    last_improve = 0
    dev_best_loss = float('inf')
    improve = '*'
    flag = False
    optimizer = torch.optim.Adam(model.parameters(),lr = config.learning_rate)

    for epoch in range(config.nums_epochs):
        print("Epoch {}/{}".format(epoch,config.nums_epochs))
        for idx,(trains,labels) in enumerate(train_iter):
            total_batch += 1
            train , _ = trains
            output = model(train)

            loss = F.cross_entropy(output,labels)
            loss.backward()
            optimizer.step()

            if total_batch % 5 == 0:
                true = labels.data.cpu()
                pred = torch.max(output,-1)[1].cpu()
                train_acc = metrics.accuracy_score(true,pred)

                dev_acc , dev_loss = evaluate(config,model,dev_iter)
                improve = ''
                if dev_loss < dev_best_loss:
                    # 存储最优模型
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(),config.model_path)
                    last_improve = total_batch
                    improve = '*'
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Train Acc: {2:>6.2%}, Val Loss: {3:>5.2}, Val Acc: {4:>6.2} Time: {5} {6}'
                print(msg.format(total_batch,loss,train_acc,dev_loss,dev_acc,time_dif,improve))
                model.train()

            if total_batch - last_improve > config.require_improvement:
                flag = True
                break

        if flag == True:
            print("No optimization for a long time, auto-stopping...")
            break


def test(config,model,test_iter):
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    start_time = time.time()
    test_acc,test_loss,report,confusion = evaluate(config,model,test_iter,True)
    msg = 'Test Loss: {:>6.2} ,Test Acc: {:>6.2%}'
    print(msg.format(test_loss,test_acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config,model,dev_iter,test = False):
    model.eval()
    total_loss = 0
    total_labels = []
    total_predic = []
    with torch.no_grad():
        for texts,labels in dev_iter:
            outputs = model(texts[0])
            loss = F.cross_entropy(outputs,labels)
            total_loss += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs,-1)[1].cpu().numpy()
            total_labels = np.append(total_labels,labels)
            total_predic = np.append(total_predic,predic)
        acc = metrics.accuracy_score(total_labels,total_predic)
        if test:
            report = metrics.classification_report(total_labels,total_predic)
            confusion = metrics.confusion_matrix(total_labels,total_predic)
            return acc,total_loss//len(dev_iter),report,confusion
        a = total_loss.item()
        b = len(dev_iter)
        return acc,a/b

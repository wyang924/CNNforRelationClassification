# -*- coding: utf-8 -*-

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

import dataset
import models
from config import opt


# 获取当前系统时间
def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def test(**kwargs):
    pass


def train(**kwargs):
    # Step 1：根据命令行参数更新配置
    kwargs = {'model_name': 'CNN_ModelForRC', 'lr': 0.01}
    opt.parse(kwargs) # 输入参数解析
    load_model_path = os.path.join(opt.load_model_path, opt.model_name + '.pth')

    # Step 2：数据
    # loading data
    DataModel = getattr(dataset, 'SEMData') # class SEMData
    train_data = DataModel(opt.data_root, train=True) # SEMData实例化
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_data = DataModel(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    print('train data: {}; test data: {}\n'.format(len(train_data), len(test_data))) # train data: 8000; test data: 2717

    # Step 3：模型(选择CNN模型)
    model = getattr(models, 'CNN')(opt) # CNN模型实例化
    continue_with_previous = False
    if os.path.exists(load_model_path):
        while True:
            resp = input("Found an existing model with the name {}.\n\
            Do you want to:\n\
            [c]ontinue training the existing model?\n\
            [r]eplace the existing model and train a new one?\n\
            [e]xit?\n>".format(opt.model_name + '.pth'))
            resp = resp.lower().strip()
            if resp not in ('c', 'r', 'e'):
                continue
            if resp == 'e':
                sys.exit()
            elif resp == 'c':
                continue_with_previous = True
            break
    if continue_with_previous:
        print("{} Loading previous model state.".format(now()))
        model.load(load_model_path)
    else:
        print("{} New model will be trained.".format(now()))

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
        model.cuda()

    # Step 4：目标函数和优化器
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr)

    # Step 5：统计指标
    best_acc = 0.0

    # Step 6：train
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        # print(len(list(enumerate(train_data_loader))[0]))
        for ii, data in enumerate(train_data_loader): # data : [lexical_feature, word_feautre, left_pf, right_pf, labels]
            # torch.IntTensor converted to torch.LongTensor
            data = [e.long() for e in data]

            if opt.use_gpu:
                data = list(map(lambda x: Variable(x.cuda()), data))
            else:
                data = list(map(Variable, data))

            model.zero_grad() # torch.nn.Module.zero_grad ## Sets gradients of all model parameters to zero
            out = model(data[:-1]) # 前向传播计算，调用__call__函数中的result = self.forward(*input, **kwargs)
            loss = criterion(out, data[-1])
            loss.backward()
            optimizer.step() # Performs a single optimization step
            total_loss += loss.data.item()

        # Step 6：每个epoch结束，评估在测试集上的指标
        train_avg_loss = total_loss / len(train_data_loader.dataset)
        acc, f1, eval_avg_loss, pred_y = eval(model, test_data_loader, opt.rel_num)
        if best_acc < acc:
            best_acc = acc
            write_result(model.model_name, pred_y)
            model.save(name=opt.model_name)
        # toy_acc, toy_f1, toy_loss = eval(model, train_data_loader, opt.rel_num)
        print('{} Epoch {}/{}: train loss: {:.4f}; test accuracy: {:.4f}, test f1:{:.4f},  test loss {:.4f}'.format(
            now(), epoch+1, opt.num_epochs, train_avg_loss, acc, f1, eval_avg_loss))

    print("*" * 30)
    print("the best acc: {:.4f};".format(best_acc))


def eval(model, test_data_loader, k):
    model.eval() # 让model变成测试模式
    avg_loss = 0.0
    pred_y = []
    labels = []
    for ii, data in enumerate(test_data_loader):
        # torch.IntTensor converted to torch.LongTensor
        data = [e.long() for e in data]

        if opt.use_gpu:
            data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
        else:
            data = list(map(lambda x: torch.LongTensor(x), data))

        out = model(data[:-1])
        loss = F.cross_entropy(out, data[-1]) # torch.nn.functional.cross_entropy()
                                              # This criterion combines log_softmax and nll_loss in a single function.
        pred_y.extend(torch.max(out, 1)[1].data.cpu().numpy().tolist())
        labels.extend(data[-1].data.cpu().numpy().tolist())
        avg_loss += loss.data.item()

    size = len(test_data_loader.dataset)
    assert len(pred_y) == size and len(labels) == size
    f1 = f1_score(labels, pred_y, average='micro') # 计算F1值
    acc = accuracy_score(labels, pred_y) # 准确率
    model.train() # 让model变成训练模式
    return acc, f1, avg_loss / size, pred_y


def write_result(model_name, pred_y):
    out = open('./semeval/sem_{}_result.txt'.format(model_name), 'w')
    size = len(pred_y)
    start = 8001
    end = start + size
    for i in range(start, end):
        out.write("{}\t{}\n".format(i, pred_y[i - start]))


if __name__ == "__main__":
    # import fire
    # fire.Fire(train)
    train()

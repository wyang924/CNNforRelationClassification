# -*- coding: utf-8 -*-

from .BasicModule import BasicModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(BasicModule):
    """
    the basic model
    Zeng 2014 "Relation Classification via Convolutional Deep Neural Network"
    """

    def __init__(self, opt):
        super(CNN, self).__init__()

        self.opt = opt # 参数配置项实例
        self.model_name = opt.model_name # 模型名称

        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # word_dim = 50
        self.pos1_embs = nn.Embedding(self.opt.pos_size + 1, self.opt.pos_dim) # pos_dim = 5
        self.pos2_embs = nn.Embedding(self.opt.pos_size + 1, self.opt.pos_dim) # pos_dim = 5

        feature_dim = self.opt.word_dim + self.opt.pos_dim * 2 # feature_dim = 60

        # encoding sentence level feature via cnn
        # Conv2d(1,230,kernel_size=(3,60),stride=(1,1),pading=(1,0))
        self.convs = nn.ModuleList([nn.Conv2d(1, self.opt.filters_num, (k, feature_dim),
                                              padding=(int(k / 2), 0)) for k in self.opt.filters])
        all_filter_num = self.opt.filters_num * len(self.opt.filters) # 200*1

        # Hidden Layer1
        self.hidden_linear1 = nn.Linear(all_filter_num + self.opt.word_dim * 6, self.opt.sen_feature_dim) # y=xA^T +b

        # Hidden Layer2
        self.hidden_linear2 = nn.Linear(self.opt.sen_feature_dim, 100) # (200,100)

        # Output Layer
        self.out_linear = nn.Linear(100, self.opt.rel_num) # (100,19)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_word_emb() # 初始化词向量
        self.init_model_weight() # 初始化网络权重

    def init_model_weight(self):
        """
        use xavier to init
        """
        # 隐藏层1线性变换权重
        nn.init.xavier_normal_(self.hidden_linear1.weight)
        nn.init.constant_(self.hidden_linear1.bias, 0.)

        # 隐藏层2线性变换权重
        nn.init.xavier_normal_(self.hidden_linear2.weight)
        nn.init.constant_(self.hidden_linear2.bias, 0.)

        # 输出层线性变换权重
        nn.init.xavier_normal_(self.out_linear.weight)
        nn.init.constant_(self.out_linear.bias, 0.)

        # 卷积层权重初始化
        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def init_word_emb(self):
        w2v = torch.from_numpy(np.load(self.opt.w2v_path))

        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
        else:
            self.word_embs.weight.data.copy_(w2v)

    def forward(self, x):

        lexical_feature, word_feautre, left_pf, right_pf = x

        # lexical word embedding
        batch_size = lexical_feature.size(0) # lexical_feature.size(1) = 6
        lexical_level_emb = self.word_embs(lexical_feature)  # (batch_size, 6, word_dim)
        lexical_level_emb = lexical_level_emb.view(batch_size, -1) # flatten the output, (batch_size, 6*word_dim)

        # sentence level feature
        word_emb = self.word_embs(word_feautre)  # (batch_size, max_len, word_dim)
        left_emb = self.pos1_embs(left_pf)  # (batch_size, max_len, word_dim)
        right_emb = self.pos2_embs(right_pf)  # (batch_size, max_len, word_dim)

        sentence_feature = torch.cat([word_emb, left_emb, right_emb], 2)  # (batch_size, max_len, word_dim + pos_dim *2)

        # conv part
        # unsqueeze 升维，Add a dimension of 1 in the 1th position, torch.Size([128,98,60]) ==> torch.Size([128,1,98,60])
        x = sentence_feature.unsqueeze(1)
        x = self.dropout(x)
        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs] # squeeze 降维，remove the 3rd position of 1
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1) # 128 * 230

        sen_level_emb = x
        # combine lexical and sentence level emb
        x = torch.cat([lexical_level_emb, sen_level_emb], 1) # cat(128 * 300 , 128 * 230) ==> 128 * 530
        x = self.dropout(x)
        x = self.hidden_linear1(x)

        x = self.dropout(x)
        x = self.hidden_linear2(x)

        x = self.dropout(x)
        x = self.out_linear(x) # 128*530 530*19 = 128 * 19

        return x

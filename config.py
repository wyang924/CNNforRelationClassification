# -*- coding: utf-8 -*-
import warnings

data_dic = {
    'SEM': {
        'data_root': './dataset/SemEval/',
        'w2v_path': './dataset/SemEval/train/npy/w2v.npy',
        # 'p2v_path': './dataset/SemEval/train/npy/p2v.npy',
        'vocab_size': 22315,  # vocab + UNK + BLANK
        'rel_num': 19
    }
}


class DefaultConfig(object):
    data = 'SEM'  # SEM
    data_root = data_dic[data]['data_root']  # the data dir
    w2v_path = data_dic[data]['w2v_path']

    model_name = 'SEM_CNN'
    load_model_path = 'checkpoints' # train model path

    batch_size = 128  # batch size
    use_gpu = False  # user GPU or not
    gpu_id = 0
    num_workers = 4  # how many workers for loading data

    vocab_size = data_dic[data]['vocab_size']  # vocab + UNK + BLANK
    rel_num = data_dic[data]['rel_num']
    word_dim = 50
    pos_dim = 5

    pos_size = 102

    num_epochs = 300  # the number of epochs for training
    drop_out = 0.5
    lr = 0.01  # initial learning rate

    # Conv
    filters = [3]  # windows size
    filters_num = 200
    sen_feature_dim = filters_num

def parse(self, kwargs):
    """
    user can update the default hyperparamter
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut {}".format(k))
        setattr(self, k, v)

    data_list = ['data_root', 'w2v_path', 'rel_num', 'vocab_size']
    for r in data_list:
        setattr(self, r, data_dic[self.data][r])

    print('*************************************************')
    print('user config:')
    for k, v in kwargs.items():
        if not k.startswith('__'):
            print("{} => {}".format(k, getattr(self, k)))
    print('*************************************************')


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
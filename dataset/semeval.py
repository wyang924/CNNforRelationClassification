# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import numpy as np
import string
from nltk.corpus import wordnet

# 数据集：SemEval2010_task8_all_data
# Hendrickx et al.,2010 Semeval-2010 task 8: Multi-way classification of semantic relations between pairs of nominals
class SEMData(Dataset):

    def __init__(self, root_path, train=True):
        if train:
            path = os.path.join(root_path, 'train/npy/')
            print('loading train data')
        else:
            path = os.path.join(root_path, 'test/npy/')
            print('loading test data')

        self.word_feautre = np.load(path + 'word_feautre.npy')
        self.lexical_feature = np.load(path + 'lexical_feature.npy')
        self.right_pf = np.load(path + 'right_pf.npy')
        self.left_pf = np.load(path + 'left_pf.npy')
        self.labels = np.load(path + 'labels.npy')
        self.x = list(zip(self.lexical_feature, self.word_feautre, self.left_pf, self.right_pf, self.labels)) # 特征拼接([lexical_feature],[word_feature],[left_pf],[right_pf],[labels])
        print('loading finish')

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)


class SEMLoad(object):
    """
    load and preprocess data
    """
    def __init__(self, root_path, train=True, max_len=98, limit=50):

        self.stoplists = set(string.punctuation)

        self.max_len = max_len
        self.limit = limit
        self.root_path = root_path
        self.train = train
        if self.train:
            print('train data:')
        else:
            print('test data:')

        self.hypernymsnum = 1  # 最多获取5个上位词

        self.rel_path = os.path.join(root_path, 'relation2id.txt')
        self.w2v_path = os.path.join(root_path, 'vector_50.txt')
        self.train_path = os.path.join(root_path, 'train.txt')
        self.vocab_path = os.path.join(root_path, 'vocab.txt')
        self.test_path = os.path.join(root_path, 'test.txt')

        print('loading start....')
        self.rel2id, self.id2rel = self.load_rel()
        self.w2v, self.word2id, self.id2word = self.load_w2v()

        if train:
            self.lexical_feature, sen_feature, self.labels = self.parse_sen(self.train_path)
        else:
            self.lexical_feature, sen_feature, self.labels = self.parse_sen(self.test_path)

        self.word_feautre, self.left_pf, self.right_pf = sen_feature # 依次为词特征WF，词位置特征PF1，词位置特征PF2
        print('loading finish')

    def save(self):
        """
        save different features as npy files
        """
        if self.train:
            prefix = 'train'
        else:
            prefix = 'test'
        np.save(os.path.join(self.root_path, prefix, 'npy/word_feautre.npy'), self.word_feautre)
        np.save(os.path.join(self.root_path, prefix, 'npy/left_pf.npy'), self.left_pf)
        np.save(os.path.join(self.root_path, prefix, 'npy/right_pf.npy'), self.right_pf)
        np.save(os.path.join(self.root_path, prefix, 'npy/lexical_feature.npy'), self.lexical_feature)
        np.save(os.path.join(self.root_path, prefix, 'npy/labels.npy'), self.labels)
        np.save(os.path.join(self.root_path, prefix, 'npy/w2v.npy'), self.w2v)
        print('save finish!')

    def load_rel(self):
        """
        load relations
        """
        rels = [i.strip('\n').split() for i in open(self.rel_path)]
        rel2id = {j: int(i) for i, j in rels}
        id2rel = {int(i): j for i, j in rels}

        return rel2id, id2rel

    def load_w2v(self):
        """
        reading from vec.bin
        add two extra tokens:
            : UNK for unkown tokens
            : BLANK for the max len sentence
        """
        wordlist = []
        vecs = []

        w2v = open(self.w2v_path)
        for line in w2v:
            line = line.strip('\n').split()
            word = line[0]
            vec = list(map(float, line[1:]))
            wordlist.append(word)
            vecs.append(np.array(vec))

        # wordlist.append('UNK')
        # wordlist.append('BLANK')
        # vecs.append(np.random.normal(size=dim, loc=0, scale=0.05))
        # vecs.append(np.random.normal(size=dim, loc=0, scale=0.05))
        # vecs.append(np.zeros(dim))
        # vecs.append(np.zeros(dim))
        word2id = {j: i for i, j in enumerate(wordlist)}
        id2word = {i: j for i, j in enumerate(wordlist)}

        return np.array(vecs, dtype=np.float32), word2id, id2word

    def parse_sen(self, path):
        """
        parse the records in data
        input：
            @param path: train/val/test data path
             e.g. 3 12 12 15 15 the system as described above ... configuration of antenna elements
        output:
            @param lexical_feature: [[id1,id2,...,id6],...,[id1,id2,...,id6]]
            @param sen_feature: [[sen_list, pos_left, pos_right],...,[sen_list, pos_left, pos_right]]
            @param all_labels: [label1,...,labeln]
        """
        all_sens = []
        all_labels = []
        for line in open(path, 'r'):
            line = line.strip('\n').split(' ')
            sens = line[5:]
            rel = int(line[0])

            ent1 = (int(line[1]), int(line[2]))
            ent2 = (int(line[3]), int(line[4]))

            all_labels.append(rel)
            sens = list(map(lambda x: self.word2id.get(x, self.word2id['<PAD>']), sens)) # word transform to id

            all_sens.append((ent1, ent2, sens)) # [((12,12),(15,15),[22,3,4,5,6]),...,]

        lexical_feature = self.get_lexical_feature(all_sens) # 词特征 [[id1,id2,...,id6],...,[id1,id2,...,id6]]
        sen_feature = self.get_sentence_feature(all_sens)    # 句子特征 [[sen_list, pos_left, pos_right],...,[sen_list, pos_left, pos_right]]

        return lexical_feature, sen_feature, all_labels

    def get_lexical_feature(self, sens):
        """
        input:
            @param sens: [[id1,...,idn],...,[id1,...,idn]]
        output:
            @param lexical_feature: [[id1,id2,...,id6],...,[id1,id2,...,id6]]
        : L1: noun1
        : L2: noun2
        : L3: left and right tokens of noun1
        : L4: left and right tokens of noun2
        : L5: WordNet hypernyms
        """

        lexical_feature = []
        for idx, sen in enumerate(sens):
            pos_e1, pos_e2, sen = sen
            left_e1 = self.get_left_word(pos_e1, sen)
            left_e2 = self.get_left_word(pos_e2, sen)
            right_e1 = self.get_right_word(pos_e1, sen)
            right_e2 = self.get_right_word(pos_e2, sen)
            hypernyms_e1 = self.get_hypernyms(pos_e1, sen) # id_list
            hypernyms_e2 = self.get_hypernyms(pos_e2, sen) # id_list

            # not add WordNet hypernyms of nouns
            # lexical_feature.append([sen[pos_e1[0]], left_e1, right_e1, \
            #                         sen[pos_e2[0]], left_e2, right_e2])

            lexical_feature.append([sen[pos_e1[0]], sen[pos_e2[0]], left_e1, \
                                    right_e1, left_e2, right_e2])

            # # add WordNet hypernyms of nouns
            # lexical_feature_e1 = [sen[pos_e1[0]], left_e1, right_e1]; lexical_feature_e2 = [sen[pos_e2[0]], left_e2, right_e2]
            # lexical_feature_e1 += hypernyms_e1
            # lexical_feature_e2 += hypernyms_e2
            #
            # lexical_feature.append([lexical_feature_e1 + lexical_feature_e2])

        return lexical_feature

    def get_sentence_feature(self, sens):
        """
        input:
            @param sens: [[id1,...,idn],...,[id1,...,idn]]
        output:
            @param sen_feature: [[sen_list, pos_left, pos_right],...,[sen_list, pos_left, pos_right]]
            sen_list: dim = max_len(98)
            pos_left: dim = max_len(98)
            pos_right: dim = max_len(98)
            return : <zip object>
        others:
            word embedding (WF)
            postion embedding (PF)
        """
        update_sens = []

        for sen in sens:
            pos_e1, pos_e2, sen = sen
            pos_left = []
            pos_right = []
            ori_len = len(sen)
            for idx in range(ori_len): # position features
                p1 = self.get_pos_feature(idx - pos_e1[0]) # 当前词与e1的相对位置
                p2 = self.get_pos_feature(idx - pos_e2[0]) # 当前词语e2的相对位置
                pos_left.append(p1)
                pos_right.append(p2)

            if ori_len > self.max_len: # 超过最大长度的sentence处理
                sen = sen[: self.max_len] # sentence截断
                pos_left = pos_left[: self.max_len]
                pos_right = pos_right[: self.max_len]
            elif ori_len < self.max_len: # 未超过最大长度的sentence进行padding处理
                sen.extend([self.word2id['<PAD>']] * (self.max_len - ori_len))
                pos_left.extend([self.limit * 2 + 2] * (self.max_len - ori_len))
                pos_right.extend([self.limit * 2 + 2] * (self.max_len - ori_len))

            update_sens.append([sen, pos_left, pos_right]) # sen,pos_left,pos_right维度相同

        return zip(*update_sens)

    def get_left_word(self, pos, sen):
        """
        get the left word id of the token of position
        """
        pos = pos[0]
        if pos > 0:
            return sen[pos - 1]
        else:
            # return sen[pos]
            return self.word2id['<PAD>']

    def get_right_word(self, pos, sen):
        """
        get the right word id of the token of position
        """
        pos = pos[1]
        if pos < len(sen) - 1:
            return sen[pos + 1]
        else:
            # return sen[pos]
            return self.word2id['<PAD>']

    def get_hypernyms(self, pos, sen):
        """
        get the hypernyms of noun1 and noun2 through WordNet
        only get 5 WordNet hypernyms at most
        """
        pos = pos[0]
        word = self.id2word[sen[pos]]
        simwords = wordnet.synsets(word, pos = wordnet.NOUN) # 获取同义词集合(仅限名词)

        res = []
        for ele in simwords:
            if word in ele.name():
                tmp = ele.hypernyms()
                res.extend(tmp)

        hypernyms = [] # 上位词
        for ele in res:
            word = ele.name().split('.')[0]
            id = self.word2id[word] if word in self.word2id.keys() else self.word2id['<PAD>']

            if id in hypernyms:
                pass
            else:
                hypernyms.append(id)

        if len(hypernyms) < self.hypernymsnum:
            hypernyms.extend([self.word2id['<PAD>']]*(self.hypernymsnum-len(hypernyms)))
        else:
            hypernyms = hypernyms[:self.hypernymsnum]
        return hypernyms

    def get_pos_feature(self, x):
        """
        clip the postion range:
        : -limit ~ limit => 0 ~ limit * 2+2
        : -51 => 0
        : -50 => 1
        : 50 => 101
        : >50: 102
        """
        if x < -self.limit:
            return 0
        if -self.limit <= x <= self.limit:
            return x + self.limit + 1
        if x > self.limit:
            return self.limit * 2 + 2


if __name__ == "__main__":
    # data = SEMLoad('./dataset/SemEval/', train=True)
    # print(len(data.word2id))
    # data.save()
    # data = SEMLoad('./dataset/SemEval/', train=False)
    # data.save()
    ## 训练数据与测试数据集预处理
    data = SEMLoad('./SemEval/', train=True)
    print(len(data.word2id))
    data.save()
    data = SEMLoad('./SemEval/', train=False)
    data.save()

import os
from random import shuffle
import pandas as pd
import numpy as np
import re
from types import MethodType, FunctionType
import jieba

import fasttext.FastText as fasttext
# import fasttext






def clean_txt(raw):
    fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
    return fil.sub(' ', raw)

def seg(sentence, sw, apply=None):
    if isinstance(apply, FunctionType) or isinstance(apply, MethodType):
        sentence = apply(sentence)
    return ' '.join([i for i in jieba.cut(sentence) if i.strip() and i not in sw])

def stop_words():
    with open('stop_words.txt', 'r', encoding='utf-8') as swf:
        return [line.strip() for line in swf]

class _MD(object):
    mapper = {
        str: '',
        int: 0,
        list: list,
        dict: dict,
        set: set,
        bool: False,
        float: .0
    }

    def __init__(self, obj, default=None):
        self.dict = {}
        assert obj in self.mapper, \
            'got a error type'
        self.t = obj
        if default is None:
            return
        assert isinstance(default, obj), \
            f'default ({default}) must be {obj}'
        self.v = default

    def __setitem__(self, key, value):
        self.dict[key] = value


    def __getitem__(self, item):
        if item not in self.dict and hasattr(self, 'v'):
            self.dict[item] = self.v
            return self.v
        elif item not in self.dict:
            if callable(self.mapper[self.t]):
                self.dict[item] = self.mapper[self.t]()
            else:
                self.dict[item] = self.mapper[self.t]
            return self.dict[item]
        return self.dict[item]


def defaultdict(obj, default=None):
    return _MD(obj, default)


class TransformData(object):
    def to_csv(self, handler, output, index=False):
        dd = defaultdict(list)
        for line in handler:
            label, content = line.split(',', 1)
            dd[label.strip('__label__').strip()].append(content.strip())

        df = pd.DataFrame()
        for key in dd.dict:
            col = pd.Series(dd[key], name=key)
            df = pd.concat([df, col], axis=1)
        return df.to_csv(output, index=index, encoding='utf-8')


def split_train_test(source, auth_data=False):
    if not auth_data:
        train_proportion = 0.8
    else:
        train_proportion = 0.7

    basename = source.rsplit('.', 1)[0]
    train_file = basename + '_train.txt'
    test_file = basename + '_test.txt'

    handel = pd.read_csv(source, index_col=False, low_memory=False)
    train_data_set = []
    test_data_set = []
    for head in list(handel.head()):
        train_num = int(handel[head].dropna().__len__() * train_proportion)
        sub_list = [f'__label__{head} , {item.strip()}\n' for item in handel[head].dropna().tolist()]
        train_data_set.extend(sub_list[:train_num])
        test_data_set.extend(sub_list[train_num:])
    shuffle(train_data_set)
    shuffle(test_data_set)

    with open(train_file, 'w', encoding='utf-8') as trainf,\
        open(test_file, 'w', encoding='utf-8') as testf:
        for tds in train_data_set:
            trainf.write(tds)
        for i in test_data_set:
            testf.write(i)

    return train_file, test_file

# 转化成csv
td = TransformData()
handler = open('data-xk.txt',"r",encoding="utf-8")
td.to_csv(handler, 'data.csv')
handler.close()

# 将csv文件切割，会生成两个文件（data_train.txt和data_test.txt）
train_file, test_file = split_train_test('data.csv', auth_data=True)

def train_model(ipt=None, opt=None, model='', dim=100, epoch=5, lr=0.1, loss='softmax'):
    np.set_printoptions(suppress=True)
    if os.path.isfile(model):
        classifier = fasttext.load_model(model)
    else:
        classifier = fasttext.train_supervised(ipt, label='__label__', dim=dim, epoch=epoch,
                                         lr=lr, wordNgrams=2, loss=loss)
        """
          训练一个监督模型, 返回一个模型对象

          @param input:           训练数据文件路径
          @param lr:              学习率
          @param dim:             向量维度
          @param ws:              cbow模型时使用
          @param epoch:           次数
          @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
          @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
          @param minn:            构造subword时最小char个数
          @param maxn:            构造subword时最大char个数
          @param neg:             负采样
          @param wordNgrams:      n-gram个数
          @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
          @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
          @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
          @param lrUpdateRate:    学习率更新
          @param t:               负采样阈值
          @param label:           类别前缀
          @param verbose:         ??
          @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
          @return model object
        """
        classifier.save_model(opt)
    return classifier

dim = 100
lr = 5
epoch = 5
model = f'data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'

classifier = train_model(ipt='data_train.txt',
                         opt=model,
                         model=model,
                         dim=dim, epoch=epoch, lr=0.5
                         )

result = classifier.test('data_test.txt')
print(result)

def cal_precision_and_recall(file='data_test.txt'):
    precision = defaultdict(int, 1)
    recall = defaultdict(int, 1)
    total = defaultdict(int, 1)
    with open(file,"r",encoding="utf-8") as f:
        for line in f:
            label, content = line.split(',', 1)
            total[label.strip().strip('__label__')] += 1
            labels2 = classifier.predict([seg(sentence=content.strip(), sw='', apply=clean_txt)])
            pre_label, sim = labels2[0][0][0], labels2[1][0][0]
            recall[pre_label.strip().strip('__label__')] += 1

            if label.strip() == pre_label.strip():
                precision[label.strip().strip('__label__')] += 1

    print('precision', precision.dict)
    print('recall', recall.dict)
    print('total', total.dict)
    for sub in precision.dict:
        pre = precision[sub] / total[sub]
        rec =  precision[sub] / recall[sub]
        F1 = (2 * pre * rec) / (pre + rec)
        print(f"{sub.strip('__label__')}  precision: {str(pre)}  recall: {str(rec)}  F1: {str(F1)}")
		
def main(source):
    basename = source.rsplit('.', 1)[0]
    csv_file = basename + '.csv'

    td = TransformData()
    handler = open(source,"r",encoding="utf-8")
    td.to_csv(handler, csv_file)
    handler.close()

    train_file, test_file = split_train_test(csv_file)

    dim = 100
    lr = 5
    epoch = 5
    model = f'data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'

    classifier = train_model(ipt=train_file,
                             opt=model,
                             model=model,
                             dim=dim, epoch=epoch, lr=0.5
                             )

    result = classifier.test(test_file)
    print(result)

    cal_precision_and_recall(test_file)


if __name__ == '__main__':
    main('data-xk.txt')
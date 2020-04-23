#-*- coding:utf-8 -*-


import jieba
import random
import fasttext.FastText as ff
import os
import random
from random import shuffle

category_list=os.listdir("data/1340315556638976_1/")
train_data=[]
test_data=[]

#
with open("train.txt","w",encoding="utf-8") as train_file, open("test.txt","w",encoding="utf-8") as test_file:
    for category in category_list:
        with open(f"data/1340315556638976_1/{category}","r",encoding="utf-8") as read_file:
            for line in read_file.readlines():
                sentence=line.strip()
                content=jieba.cut(sentence)
                content = ' '.join(content)
                label="__label__"+category[:-4]

                write_sentnce=content+"\t"+label
                if random.random()>0.3:
                    train_data.append(write_sentnce)
                else:
                    test_data.append(write_sentnce)
    shuffle(train_data)
    shuffle(test_data)

    for line in train_data:
        train_file.write(line+"\n")
    for line in test_data:
        test_file.write(line+"\n")

# with open('../data/train_data.txt','r',encoding='utf-8') as file ,open('train.txt','w',encoding='utf-8') as trainfile,\
#     open('text.txt','w',encoding='utf-8') as testfile:
#     for line in file.readlines():
#         label,content=line.strip().split('\t')[1],line.strip().split('\t')[0]
#         content=jieba.cut(content)
#         content=' '.join(content)
#         label="__label__"+label
#         if random.random()>0.2:
#             trainfile.write(content.strip()+"\t"+label+'\n')
#         else:
#             testfile.write(content.strip()+"\t"+label+'\n')



#训练监督文本，train_data.txt，模型会默认保存在当前目录下，名称为"fasttext_test.model.bin"；thread表示以3个线程进行训练，不加默认1个线程
classifier = ff.train_supervised('train.txt',label='__label__',epoch=10)

# 验证数据集
result = classifier.test('test.txt')
# 输出准确率和召回率
print(result[1], result[2])
# 预测文本分类, articles是一段文本用字符串表示, k=3表示输入可能性较高的三个分类，不加参数默认只输出一个
result = classifier.predict("我 明天 去 北京 出差 支持 客户", k=3)
print(result)

# with open("test.txt","r",encoding="utf-8") as testfile:
#     for line in testfile.readlines():
#         content,label=line.strip().split("\t")
#         predit_label=classifier.predict(content)
#         print("预测值为：",predit_label,"实际值为：",label)

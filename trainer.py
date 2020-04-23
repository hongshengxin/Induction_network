import torch
from torch import nn
from time import time
from torch.autograd import Variable
import numpy as np

from Util import few_data, device
from task_generator import OmniglotTask, get_data_loader
import random
import Constants


def train(feature_encoder, relation_network, train_data, config):
    task = OmniglotTask(train_data, config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"], config["BATCH_NUM_PER_CLASS"],
                        "train")
    sample_dataloader = get_data_loader(task, config, num_per_class=config["SAMPLE_NUM_PER_CLASS"], split="train",
                                        shuffle=False)
    batch_dataloader = get_data_loader(task, config, num_per_class=config["BATCH_NUM_PER_CLASS"], split="test",
                                       shuffle=True)
    # sample datas
    samples, sample_labels, class_folders = sample_dataloader.__iter__().next()
    batches, batch_labels, class_folders = batch_dataloader.__iter__().next()

    # calculate features
    sample_features0 = feature_encoder(Variable(samples)).to(device)  # 25*128
    sample_features1 = sample_features0.view(config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"], -1)
    sample_features = torch.sum(sample_features1, 1).squeeze(1)
    batch_features = feature_encoder(Variable(batches)).to(device)  # 75*300

    # calculate relations
    sample_features_ext = sample_features.unsqueeze(0).repeat(config["BATCH_NUM_PER_CLASS"] * config["CLASS_NUM"], 1, 1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(config["CLASS_NUM"], 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

    relations = relation_network(sample_features_ext, batch_features_ext).view(-1, config["CLASS_NUM"])

    mse = nn.MSELoss().to(device)
    batch_labels = batch_labels.long()
    one_hot_labels = torch.zeros(config["BATCH_NUM_PER_CLASS"] * config["CLASS_NUM"], config["CLASS_NUM"])
    one_hot_labels = one_hot_labels.scatter_(1, batch_labels.view(-1, 1), 1)
    one_hot_labels = Variable(one_hot_labels).to(device)

    loss = mse(relations, one_hot_labels)
    feature_encoder.zero_grad()
    relation_network.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)

    return loss.item()


def valid(feature_encoder, relation_network, test_data, config, word2index):
    # test
    feature_encoder.eval()
    relation_network.eval()

    total_rewards = 0

    for i in range(config["TEST_EPISODE"]):  # 训练测试 集合数量不同
        task = OmniglotTask(test_data, config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"],
                            config["BATCH_NUM_PER_CLASS"], "test")
        sample_dataloader = get_data_loader(task, config, num_per_class=config["SAMPLE_NUM_PER_CLASS"], split="train",
                                            shuffle=False)
        test_dataloader = get_data_loader(task, config, num_per_class=config["SAMPLE_NUM_PER_CLASS"], split="test",
                                          shuffle=True)

        sample_images, sample_labels, class_folders = sample_dataloader.__iter__().next()
        test_images, test_labels, class_folders = test_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(sample_images).to(device))  # 5x28->   #5*50
        sample_features = sample_features.view(config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"], -1)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        test_features = feature_encoder(Variable(test_images).to(device))  # 20x64

        sample_features_ext = sample_features.unsqueeze(0).repeat(config["SAMPLE_NUM_PER_CLASS"] * config["CLASS_NUM"],
                                                                  1, 1)
        test_features_ext = test_features.unsqueeze(0).repeat(config["CLASS_NUM"], 1, 1)
        test_features_ext = torch.transpose(test_features_ext, 0, 1)

        relations = relation_network(sample_features_ext, test_features_ext)  # 25
        relations = relations.view(-1, config["CLASS_NUM"])  # 5*5

        _, predict_labels = torch.max(relations.data, 1)
        predict_labels = predict_labels.cpu()
        rewards = [1 if int(predict_labels[j]) == int(test_labels[j]) else 0 for j in
                   range(config["CLASS_NUM"] * config["SAMPLE_NUM_PER_CLASS"])]
        if i % 10 == 0:
            pass
            # print("测试集，目标值：{}，预测结果：{}".format(test_labels,predict_labels))
        if i % 100 == 0:
            sentence = "目的地改为哈尔滨"
            # task = OmniglotTask(test_data, config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"],
            #                     config["BATCH_NUM_PER_CLASS"], "test")
            keys = test_data.keys()
            support_inputs = []
            choice_num=config["CLASS_NUM"]*config["SAMPLE_NUM_PER_CLASS"]//len(keys)
            for categeory in keys:
                class_folders = random.sample(test_data[categeory], choice_num)
                for sentence_i in class_folders:
                    sentence_index = sentence2indices(sentence_i, word2index, config["max_len"], Constants.PAD)
                    support_inputs.append(sentence_index)
            # support_inputs.extend([support_inputs[-1]*(config["CLASS_NUM"]*config["SAMPLE_NUM_PER_CLASS"]-len(support_inputs))])
            # sample_dataloader = get_data_loader(task, config, num_per_class=config["SAMPLE_NUM_PER_CLASS"],
            #                                     split="train",
            #                                     shuffle=False)
            # sample_images, sample_labels,class_folders = sample_dataloader.__iter__().next()
            support_inputs = torch.tensor(support_inputs)
            """"dfalsfd"""
            sentence_id = [config["word2index"].get(word, Constants.PAD) for word in sentence]
            sentence_id += [0] * (12 - len(sentence_id))
            sentence_images = torch.tensor([sentence_id])
            sentence_images = sentence_images.repeat(config["CLASS_NUM"] * config["SAMPLE_NUM_PER_CLASS"], 1)
            test_images = sentence_images

            sample_features = feature_encoder(Variable(support_inputs).to(device))  # 5x28->   #5*50
            sample_features = sample_features.view(config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"], -1)
            sample_features = torch.sum(sample_features, 1).squeeze(1)
            test_features = feature_encoder(Variable(test_images).to(device))  # 20x64

            sample_features_ext = sample_features.unsqueeze(0).repeat(
                config["SAMPLE_NUM_PER_CLASS"] * config["CLASS_NUM"],
                1, 1)
            test_features_ext = test_features.unsqueeze(0).repeat(config["CLASS_NUM"], 1, 1)
            test_features_ext = torch.transpose(test_features_ext, 0, 1)

            relations = relation_network(sample_features_ext, test_features_ext)  # 25
            relations = relations.view(-1, config["CLASS_NUM"])  # 5*5

            _, predict_labels = torch.max(relations.data, 1)
            predict_labels = predict_labels.cpu()
            print("预测概率为：", predict_labels)
            print("预测值为", predict_labels, keys)

        total_rewards += np.sum(rewards)

    test_accuracy = total_rewards / 1.0 / config["CLASS_NUM"] / config["SAMPLE_NUM_PER_CLASS"] / config["TEST_EPISODE"]
    print("test accuracy:", test_accuracy)

    return test_accuracy


def sentence2indices(line, word2index, max_len=None, padding_index=None, unk=None, began=None, end=None):
    result = [word2index.get(word, unk) for word in line if word in word2index]
    if max_len is not None:
        result = result[:max_len]
    if began is not None:
        result.insert(0, began)
    if end is not None:
        result.append(end)
    if padding_index is not None and len(result) < max_len:
        result += [padding_index] * (max_len - len(result))
    if not result:
        a = 0
    # assert len(result) == max_len
    return result


def predict(feature_encoder, relation_network, config, word2index):
    '''
    预测函数
    :param feature_encoder:
    :param relation_network:
    :param config:
    :return:
    '''

    sentence = "这个月金蝶中国的收款"

    sentence_id = [word2index[word] for word in sentence]
    sentence_id += [0] * (12 - len(sentence_id))
    sample_sentence = torch.tensor(sentence_id)

    # calculate features
    sample_features = feature_encoder(Variable(sample_sentence).to(device))  # 5x28->   #5*50
    sample_features = sample_features.view(config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"], -1)
    sample_features = torch.sum(sample_features, 1).squeeze(1)

    sample_features_ext = sample_features.unsqueeze(0).repeat(config["SAMPLE_NUM_PER_CLASS"] * config["CLASS_NUM"],
                                                              1, 1)
    test_features_ext = test_features.unsqueeze(0).repeat(config["CLASS_NUM"], 1, 1)
    test_features_ext = torch.transpose(test_features_ext, 0, 1)

    relations = relation_network(sample_features_ext, test_features_ext)  # 25
    relations = relations.view(-1, config["CLASS_NUM"])  # 5*5

    _, predict_labels = torch.max(relations.data, 1)


if __name__ == "__main__":
    t0 = time()
    print("耗时", time() - t0)

import torch

from WaveLR import WaveLR
from models import RelationNetwork
from time import time
from torch.optim.lr_scheduler import StepLR
from StructuredSelfAttention import StructuredSelfAttention
from task_generator import omniglot_character_folders,train_omniglot_character_folders
from trainer import train, valid

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def main():
    train_data_ori, test_data_ori, word2index, labels = omniglot_character_folders()

    train_data=train_omniglot_character_folders("data/train",labels)
    test_data=train_omniglot_character_folders("data/1449214354962688_1",labels)
    config = {
        "CLASS_NUM": 12,
        "SAMPLE_NUM_PER_CLASS": 5,
        "BATCH_NUM_PER_CLASS": 5,
        "EPISODE": 10000,  # 1000000
        "TEST_EPISODE": 10,  # 1000
        "LEARNING_RATE": 0.0001,  # 0.01
        "FEATURE_DIM": 256,  # lstm_hid_dim *2
        "RELATION_DIM": 8,
        "use_bert": False,
        "max_len": 12,
        "emb_dim": 300,
        "lstm_hid_dim": 128,
        "d_a": 64,
        "r": 1,
        "n_classes": 5,
        "num_layers": 1,
        "dropout": 0.1,
        "type": 1,
        "use_pretrained_embeddings": True,
        "word2index": word2index,
        "vocab_size": len(word2index)
    }
    feature_encoder = StructuredSelfAttention(config).to(device)
    relation_network = RelationNetwork(2 * config["FEATURE_DIM"], config["RELATION_DIM"]).to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-4)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-4)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)
    print("开始训练")
    t0 = time()

    for episode in range(config["EPISODE"]):
        feature_encoder.train()
        relation_network.train()
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        loss = train(feature_encoder, relation_network, train_data, config)

        feature_encoder_optim.step()
        relation_network_optim.step()
        print("episode:", loss)
        if (episode + 1) % 10 == 0:
            print("episode:", episode + 1, "loss", loss, "耗时", time() - t0)
            t0 = time()

        if (episode + 1) % (config["TEST_EPISODE"]) == 0:
            test_accuracy = valid(feature_encoder, relation_network, test_data, config,word2index)
            t0 = time()
            print("\n")
            print("testting set 准确率为：",test_accuracy)
            print("\n")
    print("直接词向量")
    print("完成")


if __name__ == "__main__":
    t0 = time()
    main()
    print("耗时", time() - t0)

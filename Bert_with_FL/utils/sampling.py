#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

# from Bert_finetune import Dataset
from transformers import BertTokenizer


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def bbc_iid(dataset, num_users):
    """
    Sample I.I.D. client data from bbc news dataset
    :param dataset:bbc-text.csv
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # extract our labels from the df
        self.labels = [labels[label] for label in df["category"]]
        self.labels = torch.Tensor(self.labels).long()  # todo
        # tokenize our texts to the format that BERT expects to get as input
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for
                      text in df["text"]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    # fetch a batch of labels
    def get_batch_labels(self, indx):
        return np.array(self.labels[indx])

    # fetch a batch of texts
    def get_batch_texts(self, indx):
        return self.texts[indx]

    # get an item with the texts and the label
    def __getitem__(self, indx):
        batch_texts = self.get_batch_texts(indx)
        batch_y = self.get_batch_labels(indx)

        return batch_texts, batch_y

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    source_url = "../data/bbc-text.csv"
    df = pd.read_csv(source_url)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    labels = {
        'business': 0,
        'entertainment': 1,
        'sport': 2,
        'tech': 3,
        'politics': 4
    }

    df_train, df_valid, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])

    # creating a custom Dataset objects using the training and validation data
    train, val = Dataset(df_train), Dataset(df_valid)
    # print(val.texts)
    # print(val.labels)
    # creating dataloaders
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)  # dataset_train

    num = 100
    d1 = mnist_noniid(dataset_train, num)
    d2 = bbc_iid(train_dataloader, num)
# todo:格式不一样，
#     d1格式:{num1: array([id1,id2..], dtype=int64),num2: array([id1,id2..], dtype=int64)..}
#     d2格式：{num1:{id1,id2..},num2:{id1,id2..}}
    print(type(d1))
    print(type(d1[0]))
    print(d1[0])
    print(d2)
    print(type(d2))
    print(type(d2[0]))
    print(np.array(list(d2[0])))
    print(type(np.array(list(d2[0]))))

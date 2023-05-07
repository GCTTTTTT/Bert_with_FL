#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print("self.dataset", self.dataset)
        # print("self.dataset[self.idxs]", self.dataset[self.idxs])
        # print("self.dataset[self.idxs[item]]",self.dataset[self.idxs[item]])

        image, label = self.dataset[self.idxs[item]]
        return image, label

class DatasetSplit_BBC(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        print("self.dataset[self.idxs[item]]",self.dataset[self.idxs[item]])
        print("self.dataset[self.idxs]",self.dataset[self.idxs])
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print("111111111111111111")
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdate_Bert(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # for batch_idx, (images, labels) in enumerate(self.ldr_train):
            for batch_idx, (train_input, train_label) in enumerate(self.ldr_train):
            # for train_input, train_label in tqdm(train_dataloader):
            #     images, labels = images.to(self.args.device), labels.to(self.args.device)

                train_label = train_label.to(self.args.device)
                mask = train_input['attention_mask'].to(self.args.device)
                input_id = train_input['input_ids'].squeeze(1).to(self.args.device)
                #             print(input_id.shape)

                # get the predictions
                output = net(input_id, mask) # log_probs = net(images)


                # log_probs = net(images)
                # loss = self.loss_func(log_probs, labels)
                loss = self.loss_func(output, train_label)

                # total_loss_train += batch_loss.item()
                # acc = (output.argmax(dim=1) == train_label).sum().item()
                # total_acc_train += acc
                net.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print("111111111111111111")
                    # print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     iter, batch_idx * len(images), len(self.ldr_train.dataset),
                    #            100. * batch_idx / len(self.ldr_train), loss.item()))
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(input_id), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                    print(
                        f'Epochs: {iter} | Train Loss: {loss.item(): .3f} '
                        )
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

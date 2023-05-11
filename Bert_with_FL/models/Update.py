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
    def __init__(self, args, dataset=None, val_data=None,idxs=None,idxs_val=None):
        self.args = args
        self.ds = dataset # todo
        self.idx = idxs # todo
        self.idx_val = idxs_val # todo
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_val = DataLoader(DatasetSplit(val_data, idxs_val), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        # print("len ldr_train",len(self.ldr_train))
        # print(self.idx)
        # print("DatasetSplit(self.ds, self.idx):",DatasetSplit(self.ds, self.idx))
        # print("DatasetSplit(self.ds, self.idx) len:",DatasetSplit(self.ds, self.idx).__len__())
        # print("DatasetSplit(self.ds, self.idx) item:",DatasetSplit(self.ds, self.idx).__getitem__(next(iter(self.idx))))
        # print("DatasetSplit(self.ds, self.idx) item:",DatasetSplit(self.ds, self.idx).__getitem__(0))
        # for text_batch in iter(self.ldr_train):
        #     text = text_batch[0]
        #     text2 = text_batch
        #     print("text",text)
        #     print("text2",text2)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        epoch_acc = []

        epoch_loss_val = []
        epoch_acc_val = []

        for iter1 in range(self.args.local_ep):
            batch_loss = []
            batch_acc = []
# todo
            total_acc_train = 0
            total_loss_train = 0

            batch_loss_val = []
            batch_acc_val = []

            # for batch_idx, (images, labels) in enumerate(self.ldr_train):
            for batch_idx, (train_input, train_label) in enumerate(self.ldr_train):
            # for train_input, train_label in tqdm(train_dataloader):
            #     images, labels = images.to(self.args.device), labels.to(self.args.device)

                # print("train_input",train_input)
                # print("train_input",train_label)
                train_label = train_label.to(self.args.device)
                mask = train_input['attention_mask'].to(self.args.device)
                input_id = train_input['input_ids'].squeeze(1).to(self.args.device)
                #             print(input_id.shape)

                # get the predictions
                output = net(input_id, mask) # log_probs = net(images)


                # log_probs = net(images)
                # loss = self.loss_func(log_probs, labels)
                loss = self.loss_func(output, train_label)

                total_loss_train += loss.item() # todo

                # total_loss_train += batch_loss.item()
                acc = (output.argmax(dim=1) == train_label).sum().item() # todo
                # print("acc:",acc,"    batch_idx:",batch_idx)
                total_acc_train += acc
                # print("total_acc_train:", acc, "    batch_idx:", batch_idx)
                net.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 2 == 0:
                # if self.args.verbose:
                    # print("111111111111111111")
                    # print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     iter, batch_idx * len(images), len(self.ldr_train.dataset),
                    #            100. * batch_idx / len(self.ldr_train), loss.item()))
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter1, batch_idx * len(input_id), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                    print(
                        f'Epochs: {iter1} | Train Loss: {loss.item(): .3f} '
                        )

                batch_loss.append(loss.item())
                batch_acc.append(acc)

            with torch.no_grad():
                print("len(self.ldr_val):",len(self.ldr_val))
                # for batch_idx_val, (val_input, val_label) in enumerate(self.ldr_val):
                for i, (val_input, val_label) in enumerate(self.ldr_val):
                    val_label = val_label.to(self.args.device)
                    mask = val_input['attention_mask'].to(self.args.device)
                    input_id = val_input['input_ids'].squeeze(1).to(self.args.device)

                    output = net(input_id, mask)

                    loss_val = self.loss_func(output, val_label)
                    # total_loss_val += batch_loss.item()

                    acc_val = (output.argmax(dim=1) == val_label).sum().item()

                    # total_acc_val += acc
                    batch_loss_val.append(loss_val.item())
                    batch_acc_val.append(acc_val)

            # print("total_acc_train",total_acc_train,"  iter:",iter1)
            # print("total_acc_train / len(self.idx)",total_acc_train / len(self.idx))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print("batch_acc: ",batch_acc)
            print("len idx: ",len(self.idx))
            epoch_acc.append(sum(batch_acc)/len(self.idx))

            epoch_loss_val.append(sum(batch_loss_val) / len(batch_loss_val))
            print("batch_acc_val: ", batch_acc_val)
            print("len idx val: ", len(self.idx_val))
            epoch_acc_val.append(sum(batch_acc_val) / len(self.idx_val))

        print("epoch_acc: ",epoch_acc)
        print("sum(epoch_acc) / len(epoch_acc)： ",sum(epoch_acc) / len(epoch_acc))
        print("sum(epoch_acc_val) / len(epoch_acc_val)： ",sum(epoch_acc_val) / len(epoch_acc_val))
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),sum(epoch_acc) / len(epoch_acc) , sum(epoch_acc_val) / len(epoch_acc_val)



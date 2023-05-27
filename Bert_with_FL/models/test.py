#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        #注释了这两行
        # if args.gpu != -1:
        #     data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_Bert(net_g, datatest, args):
    # print("args.device: ",args.device)
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0

    epoch_loss = []
    epoch_acc = []

    batch_loss = []
    batch_acc = []
    # todo
    total_acc_test = 0
    total_loss_test = 0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    # print("len(data_loader): ",len(data_loader))
    # for idx, (data, target) in enumerate(data_loader):
    for batch_idx, (train_input, train_label) in enumerate(data_loader):
        #注释了这两行
        # print("batch_idx: ",batch_idx)
        # if args.gpu != -1:
        #     data, target = data.cuda(), target.cuda()


        train_label = train_label.to(args.device)
        mask = train_input['attention_mask'].to(args.device)
        input_id = train_input['input_ids'].squeeze(1).to(args.device)
        # print("train_label_device: ",train_label.device)
        # print("mask_device: ",mask.device)
        # print("input_id _device: ",input_id.device)
        # #             print(input_id.shape)
        # train_label = train_label
        # mask = train_input['attention_mask']
        # input_id = train_input['input_ids'].squeeze(1)
        #             print(input_id.shape)


        # log_probs = net_g(data)
        output = net_g(input_id, mask)


        # sum up batch loss
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # test_loss += F.cross_entropy(log_probs, train_label, reduction='sum').item() # todo
        loss = F.cross_entropy(output, train_label, reduction='sum') # todo
        total_loss_test += loss.item()

        # total_loss_train += batch_loss.item()
        acc = (output.argmax(dim=1) == train_label).sum().item()  # todo
        # print("acc:",acc,"    batch_idx:",batch_idx)
        total_acc_test += acc

        # # get the index of the max log-probability
        # y_pred = log_probs.data.max(1, keepdim=True)[1]
        # correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        batch_loss.append(loss.item())
        batch_acc.append(acc)
    # print("len(batch_loss)：",len(batch_loss))
    # print("len(data_loader.dataset)：",len(data_loader.dataset))
    # test_loss = sum(batch_loss) / len(batch_loss)
    test_loss = sum(batch_loss) / len(data_loader.dataset)
    # print("test_loss: ",test_loss)
    # print("test_loss2: ",test_loss2)
    # print("batch_acc: ",batch_acc)
    # print("len idx: ",len(self.idx))
    # accuracy = sum(batch_acc) / len(self.idx)
    accuracy = sum(batch_acc) / len(data_loader.dataset)
    # print("accuracy: ",accuracy)


    # test_loss /= len(data_loader.dataset)
    # accuracy = 100.00 * correct / len(data_loader.dataset)
    # if args.verbose:
    #     print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
    #         test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

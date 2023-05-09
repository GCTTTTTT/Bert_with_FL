#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(args,w):
    # print("w_type:", type(w))
    w_avg = copy.deepcopy(w[0])
    # w_avg.to(args.device)
    # print("w_avg_type:",type(w_avg))
    # print("w_avg_type:",type(w[1][k]))
    for k in w_avg.keys():
        # !! 将w[i][k] 和 w_avg[k]放到指定设备，否则会报错
        w_avg[k] = w_avg[k].to(args.device)
        for i in range(1, len(w)):
            # print("w_avg[k]_type:", type(w_avg[k]))
            # print("w[i][k]_type:", type(w[i][k]))
            # print("args.device:",args.device)
            # print("w[i][k]_before:",w[i][k].device.type)

            w[i][k] = w[i][k].to(args.device)
            # print("w[i][k]_after:", w[i][k].device.type)
            # print("w_avg[k] before type:", w_avg[k].device.type)

            # print("w_avg[k] after type:", w_avg[k].device.type)
            w_avg[k] += w[i][k]
            # torch.div:张量和标量做逐元素除法 或者两个可广播的张量之间做逐元素除法
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

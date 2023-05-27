#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

# FedAvg算法聚合本地权重
def FedAvg(args,w):
    # print("w_type:", type(w))
    w_avg = copy.deepcopy(w[0])
    # print("w_avg: ",w_avg)
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

# FedProx算法聚合本地权重
def FedProx(args, w_locals):
    # 获取设备类型和学习率
    # device = args['device']
    # lr = args['lr']

    # 设置超参数
    # mu = args['mu']
    w_glob = copy.deepcopy(w_locals[0])

# 将张量放到对应设备
    for k in w_glob.keys():
        w_glob[k] = w_glob[k].to(args.device)
    for i in range(len(w_locals)):
        for k in w_locals[i].keys():
            w_locals[i][k] = w_locals[i][k].to(args.device)

    # 对本地权重进行加权平均聚合，并引入正则化项
    for k in w_glob.keys():
        for i in range(1, len(w_locals)):
            w_glob[k] += w_locals[i][k]
        w_glob[k] = torch.div(w_glob[k], len(w_locals))
        # FedProx 算法引入的正则化项
        w_glob[k] += (args.mu / len(w_locals)) * (w_glob[k] - w_locals[0][k])

    #     # 将模型参数移回 CPU 设备并返回全局模型参数
    # for k in w_glob.keys():
    #     w_glob[k] = w_glob[k].cpu()
    # return w_glob

    return w_glob


def FedNova(args, w_locals, grad_locals):

    # w_locals = [w_local.to(args.device) for w_local in w_locals]


    # 初始化全局权重
    w_glob = copy.deepcopy(w_locals[0])

    for k in w_glob.keys():
        w_glob[k] = w_glob[k].to(args.device)
    for i in range(len(w_locals)):
        for k in w_locals[i].keys():
            w_locals[i][k] = w_locals[i][k].to(args.device)
    for i in range(len(grad_locals)):
        for k1 in grad_locals[i].keys():
            grad_locals[i][k1] = grad_locals[i][k1].to(args.device)

    # grad_locals = [grad_local.to(args.device) for grad_local in grad_locals]

    for key in w_glob.keys():
        w_glob[key].data.fill_(0)

    # # 计算加权平均梯度
    # weighted_grad_sum = None
    # # for i, client in enumerate(args["clients"]):
    # for i in range(args.num_users):
    #     # 获取本地梯度和权重
    #     w, grad = w_locals[i], grad_locals[i]
    #
    #     # 计算加权梯度
    #     for key in grad:
    #         # grad[key] *= w
    #         grad[key] = grad[key] * w
    #         if weighted_grad_sum is None:
    #             weighted_grad_sum = grad[key]
    #         else:
    #             weighted_grad_sum += grad[key]
    #
    # # 计算加权平均梯度
    # avg_weighted_grad = {}
    # for key, grad in weighted_grad_sum.items():
    #     # avg_weighted_grad[key] = grad / len(args["clients"])
    #     avg_weighted_grad[key] = grad / args.num_users
    #
    # # 根据FedNova算法更新全局权重
    # for key in w_glob.keys():
    #     update_grad = avg_weighted_grad[key] + args.mu * (w_glob[key] - avg_weighted_grad[key])
    #     w_glob[key].data -= args.lr * update_grad
    print("len(w_locals): ",len(w_locals))
    print("len(grad_locals): ",len(grad_locals))
# todo:test:
    weighted_grad_sum = {}
    grad = grad_locals[0]
    for i in range(args.num_users):
        # w, grad = w_locals[i], grad_locals[i]
        w = w_locals[i]
        for key, val in grad.items():
            weighted_grad = w[key] * val
            if key not in weighted_grad_sum:
                weighted_grad_sum[key] = weighted_grad
            else:
                weighted_grad_sum[key] += weighted_grad
    avg_weighted_grad = {k: v / args.num_users for k, v in weighted_grad_sum.items()}

    print("avg_weighted_grad: ",avg_weighted_grad,"============avg_weighted_grad ========")
    # 添加以下代码来检查键是否匹配
    # assert all(key in w_glob for key in avg_weighted_grad.keys()), "Keys do not match!"
    print("w_glob.keys(): ",w_glob.keys(),"=============w_glob.keys()=============")
    for key in w_glob.keys():
        # print("")
        update_grad = avg_weighted_grad[key] + args.mu * (w_glob[key] - avg_weighted_grad[key])
        w_glob[key].data -= args.lr * update_grad


    # 返回更新后的全局权重
    return w_glob
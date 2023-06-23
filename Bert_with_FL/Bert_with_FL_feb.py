import copy
import logging

import pandas as pd
import torch
from matplotlib import pyplot as plt
from transformers import BertTokenizer
import numpy as np
from transformers import BertModel
from tqdm import tqdm

from models.test import test_Bert
from models.Fed import FedAvg, FedProx, FedNova
from models.Update import LocalUpdate, LocalUpdate_Bert
from utils.sampling import bbc_iid
from utils.options import args_parser



sample = 'Hey my name is BERT'

def reset_log(log_path):
    import logging
    fileh = logging.FileHandler(log_path, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    log.setLevel(logging.DEBUG)

# def logger_config(log_path,logging_name):
#     '''
#     配置log
#     :param log_path: 输出log路径
#     :param logging_name: 记录中name，可随意
#     :return:
#     '''
#     '''
#     logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
#     '''
#     # 获取logger对象,取名
#     logger = logging.getLogger(logging_name)
#     # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
#     logger.setLevel(level=logging.DEBUG)
#     # 获取文件日志句柄并设置日志级别，第二层过滤
#     handler = logging.FileHandler(log_path, encoding='UTF-8')
#     handler.setLevel(logging.INFO)
#     # 生成并设置文件日志格式
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
#     console = logging.StreamHandler()
#     console.setLevel(logging.DEBUG)
#     # 为logger对象添加句柄
#     logger.addHandler(handler)
#     logger.addHandler(console)
#
#     return logger

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


class BertClassifier(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        # self.bert = BertModel.from_pretrained("bert-base-cased")
        self.bert = BertModel.from_pretrained("../Bert_finetune/bert_model/bert-base-cased")
        self.dropout = torch.nn.Dropout(dropout)
        # bert output a vector of size 768
        self.lin = torch.nn.Linear(768, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):
        # as output, the bert model give us first the embedding vector of all the tokens of the sequence 
        # second we get the embedding vector of the CLS token.
        # fot a classification task it's enough to use this embedding for our classifier
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.lin(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


# we are creating a standard pytorch training loop

def train(model, train_data, val_data, learning_rate, epochs=5):
    # creating a custom Dataset objects using the training and validation data
    train, val = Dataset(train_data), Dataset(val_data)
    # print(val.texts)
    # print(val.labels)
    # creating dataloaders
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    print("use_cuda: ",use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")  #todo
    print("device:",device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            # print(f"the train input : {train_input}")
            # print(f"train label : {train_label}")

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            #             print(input_id.shape)

            # get the predictions
            output = model(input_id, mask)

            # the output is a vector of 5 values (categs)
            #             print(output)
            #             print("the output shape is" ,  output.shape)
            #             print(train_label)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # updating the Gradient Descent and Backpropagation operation
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # now we evaluate on the validation data
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


if __name__ == '__main__':


    # parse args
    args = args_parser()  # 用于调用 utils 文件夹中的 option.py 函数
    # 用来选择程序运行设备，如果有 GPU 资源则调用服务器的 GPU 做运算，否则就用 CPU 运行。
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    print(args.device)
    #todo
    # logging.basicConfig(level=logging.INFO,
    #                     filename='./save/log_fed_epoch_{}_lr_{}_numUser_{}_FedAlg_{}_model_{}_detaset_{}.log'.format(
    #                         args.epochs, args.lr, args.num_users,
    #                         args.fedAlg, args.model,
    #                         args.dataset),
    #                     # encoding='utf8',
    #                     filemode='a',
    #                     # force=True,
    #                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # reset_log('./save/log_fed_epoch_{}_lr_{}_numUser_{}_FedAlg_{}_model_{}_detaset_{}.log'.format(
    #                         args.epochs, args.lr, args.num_users,
    #                         args.fedAlg, args.model,
    #                         args.dataset))
    file_log = open('./save/log_fed_epoch_{}_lr_{}_numUser_{}_FedAlg_{}_model_{}_detaset_{}.log'.format(
                            args.epochs, args.lr, args.num_users,
                            args.fedAlg, args.model,
                            args.dataset),'a',encoding='utf8')


    logger = logging.getLogger(__name__)

    source_url = "./data/bbc-text.csv"
    df = pd.read_csv(source_url)

    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    tokenizer = BertTokenizer.from_pretrained("../Bert_finetune/bert_model/bert-base-cased/vocab.txt")
    bert_input = tokenizer(sample, padding="max_length", max_length=15, truncation=True, return_tensors="pt")

    print(bert_input["input_ids"])
    print(tokenizer.decode(bert_input["input_ids"][0]))

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

    # testing
    test = Dataset(df_test)

    # print(val.texts)
    # print(val.labels)
    # creating dataloaders
    if args.dataset == 'bbc':
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True) # dataset_train
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=2) # dataset_test

        # 返回的是一个字典类型 dict_users，key值是用户 id，values是用户拥有的文本id。
        dict_users = bbc_iid(train_dataloader, args.num_users)
        dict_users_val = bbc_iid(val_dataloader, args.num_users) # todo
        # print("dict_users: ",dict_users)
    else:
        exit('Error: unrecognized dataset')

    if args.model == 'bert':
        net_glob = BertClassifier()
    else:
        exit('Error: unrecognized model')

    print(net_glob)
    # net_glob.train() 就是对网络进行训练
    net_glob.train()

    # 这一段就是 FedAvg 代码的核心了，具体逻辑就是
    # 每个迭代轮次本地更新 --> 复制参与本轮更新的 users 的所有权重 w_locals
    # --> 通过定义的 FedAvg 函数求模型参数的平均 --> 分发到每个用户进行更新
    # copy weights
    w_glob = net_glob.state_dict()

    # 创建日志记录模型结果
    # torch.save(net_glob.state_dict(),
    #            './save/bert_model_fed_{}_{}_{}_C{}_iid{}_userNum{}.bin'.format(args.dataset, args.model, args.epochs,
    #                                                                            args.frac, args.iid,
    #                                                                            args.num_users))
    # fileLog = open('./save/log_fed_epoch_{}_lr_{}_numUser_{}_FedAlg_{}_{}_{}.txt'.format(args.dataset, args.model, args.epochs,
    #                                                                            args.frac, args.iid,
    #                                                                            args.num_users))
   # logger = logger_config(log_path='./save/log_fed_epoch_{}_lr_{}_numUser_{}_FedAlg_{}_model_{}_detaset_{}.log'.format(args.epochs, args.lr, args.num_users,
    #                                                                            args.fedAlg ,args.model,
    #                                                                            args.dataset), logging_name='log_name')

    # training
    loss_train = []
    # test plt
    test_wholeData_acc_list = []
    train_wholeData_acc_list = []
    # loss_train = []
    acc_train_list = [] # todo
    acc_val_list = [] # todo
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        acc_train_locals = [] # todo
        acc_val_locals = [] # todo

        grad_locals = [] # todo:test grad_local
        if not args.all_clients:
            w_locals = []


        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            # print("idx_users: ",idxs_users)
            # print("iter , idx:",iter,idx)
            # print("dict_user[idx]: ",dict_users[idx])
            # local = LocalUpdate(args=args, dataset=train_dataloader, idxs=dict_users[idx])
            # local = LocalUpdate_Bert(args=args, dataset=train_dataloader, idxs=dict_users[idx])
            # local = LocalUpdate_Bert(args=args, dataset=train, idxs=dict_users[idx])
            local = LocalUpdate_Bert(args=args, dataset=train, val_data=val,idxs=dict_users[idx],idxs_val=dict_users_val[idx]) # todo
            # w, loss, acc_train, acc_val = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # todo:test grad_local
            w, loss, acc_train, acc_val,grad_local = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
                # todo:test grad_local
                grad_locals.append(copy.deepcopy(grad_local))
                # grad_locals[idx] = copy.deepcopy(grad_local)
            else:
                w_locals.append(copy.deepcopy(w))


            loss_locals.append(copy.deepcopy(loss))
            acc_train_locals.append(copy.deepcopy(acc_train))
            acc_val_locals.append(copy.deepcopy(acc_val))
        # update global weights
        # w_locals.to(args.device) # todo
        # print("w_locals: ",w_locals)
        if args.fedAlg == "fedavg":
            w_glob = FedAvg(args,w_locals)
        elif args.fedAlg == "fedProx":
            w_glob = FedProx(args,w_locals)
        # print("w_glob_testProx: ", w_glob_testProx,"\n=============w_glob_testProx: ===================")
        # w_glob_testNova = FedNova(args,w_locals,grad_locals) # todo:FedNova
        # print("w_glob_testNova: ",w_glob_testNova)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss and acc
        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_train_avg = sum(acc_train_locals) / len(acc_train_locals)
        acc_val_avg = sum(acc_val_locals) / len(acc_val_locals)
        print('Round {:3d}, Average loss {:.5f} , Average acc_train {:.5f} ,Average acc_val {:.5f}'.format(iter, loss_avg,acc_train_avg,acc_val_avg))
        # logger.info('Round {:3d}, Average loss {:.3f} , Average acc_train {:.3f} ,Average acc_val {:.3f}'.format(iter, loss_avg,acc_train_avg,acc_val_avg))
        # logging.info('Round {:3d}, Average loss {:.3f} , Average acc_train {:.3f} ,Average acc_val {:.3f}'.format(iter, loss_avg,acc_train_avg,acc_val_avg))
        file_log.write('Round {:3d}, Average loss {:.5f} , Average acc_train {:.5f} ,Average acc_val {:.5f}\n'.format(iter, loss_avg,acc_train_avg,acc_val_avg))
        loss_train.append(loss_avg)
        acc_train_list.append(acc_train_avg)
        acc_val_list.append(acc_val_avg)

        # 每一轮进行评估
        # testing
        net_glob.eval()
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)
        # acc_train, loss_train = test_Bert(net_glob, train, args)
        acc_train_eval, loss_train_eval = test_Bert(copy.deepcopy(net_glob).to(args.device), train, args)
        # acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # acc_test, loss_test = test_Bert(net_glob, test, args)
        acc_test_eval, loss_test_eval = test_Bert(copy.deepcopy(net_glob).to(args.device), test, args)
        print("Round {:3d},Training accuracy: {:.5f}".format(iter, acc_train_eval))
        # logger.info("Round {:3d},Training accuracy: {:.2f}".format(iter, acc_train_eval))
        file_log.write("Round {:3d},Training accuracy: {:.5f}\n".format(iter, acc_train_eval))
        # logging.info("Round {:3d},Training accuracy: {:.2f}".format(iter, acc_train_eval))
        print("Round {:3d},Testing accuracy: {:.5f}".format(iter, acc_test_eval))
        # logger.info("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_test_eval))
        file_log.write("Round {:3d},Testing accuracy: {:.5f}\n".format(iter, acc_test_eval))
        # logging.info("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_test_eval))

        test_wholeData_acc_list.append(acc_test_eval)
        train_wholeData_acc_list.append(acc_train_eval)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train, label='loss')

    plt.plot(range(len(test_wholeData_acc_list)), test_wholeData_acc_list, label='test accuracy')
    plt.plot(range(len(train_wholeData_acc_list)), train_wholeData_acc_list, label='train accuracy')

    # 添加标题和图例
    plt.title('accuracy and loss tendency')
    plt.legend()

    plt.xlabel('epoch_num')
    # plt.ylabel('train_loss')
    plt.savefig(
        './save/fed_dataset_{}_model_{}_epoch_{}_lr_{}_iid{}_userNum{}.png'.format(args.dataset, args.model, args.epochs, args.lr, args.iid,args.num_users))


# todo
    # testing
    net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_train, loss_train = test_Bert(net_glob, train, args)
    acc_train_eval_gb, loss_train_eval_gb = test_Bert(copy.deepcopy(net_glob).to(args.device), train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # acc_test, loss_test = test_Bert(net_glob, test, args)
    acc_test_eval_gb, loss_test_eval_gb = test_Bert(copy.deepcopy(net_glob).to(args.device), test, args)
    print("Global Training accuracy: {:.5f}".format(acc_train_eval_gb))
    # logger.info("Global Training accuracy: {:.2f}".format(acc_train_eval_gb))
    file_log.write("Global Training accuracy: {:.5f}\n".format(acc_train_eval_gb))
    # logging.info("Global Training accuracy: {:.2f}".format(acc_train_eval_gb))
    print("Global Testing accuracy: {:.5f}".format(acc_test_eval_gb))
    # logger.info("Global Testing accuracy: {:.2f}".format(acc_test_eval_gb))
    file_log.write("Global Testing accuracy: {:.5f}\n".format(acc_test_eval_gb))
    # logging.info("Global Testing accuracy: {:.2f}".format(acc_test_eval_gb))

# 保存模型
#     torch.save(net_glob.state_dict(), 'bert_model.bin'
    torch.save(net_glob.state_dict(), './save/bert_model_{}_{}_{}_{}_lr_{}_iid{}_userNum{}.bin'.format(args.fedAlg,args.dataset, args.model, args.epochs, args.lr, args.iid,
                                                         args.num_users))

    file_log.close()
    # EPOCHS = 5
    # model = BertClassifier()
    # learning_rate = 1e-6
    # train(model, df_train, df_valid, learning_rate, EPOCHS)
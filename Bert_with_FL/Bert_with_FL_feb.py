import copy

import pandas as pd
import torch
from transformers import BertTokenizer
import numpy as np
from transformers import BertModel
from tqdm import tqdm

from models.Fed import FedAvg
from models.Update import LocalUpdate, LocalUpdate_Bert
from utils.sampling import bbc_iid
from utils.options import args_parser

sample = 'Hey my name is BERT'


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
    # print(val.texts)
    # print(val.labels)
    # creating dataloaders
    if args.dataset == 'bbc':
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True) # dataset_train
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=2) # dataset_test

        # 返回的是一个字典类型 dict_users，key值是用户 id，values是用户拥有的文本id。
        dict_users = bbc_iid(train_dataloader, args.num_users)
        dict_users_val = bbc_iid(val_dataloader, args.num_users) # todo
        print("dict_users: ",dict_users)
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

    # training
    loss_train = []
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
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            print("idx_users: ",idxs_users)
            print("iter , idx:",iter,idx)
            # local = LocalUpdate(args=args, dataset=train_dataloader, idxs=dict_users[idx])
            # local = LocalUpdate_Bert(args=args, dataset=train_dataloader, idxs=dict_users[idx])
            # local = LocalUpdate_Bert(args=args, dataset=train, idxs=dict_users[idx])
            local = LocalUpdate_Bert(args=args, dataset=train, val_data=val,idxs=dict_users[idx],idxs_val=dict_users_val[idx]) # todo
            w, loss, acc_train, acc_val = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            acc_train_locals.append(copy.deepcopy(acc_train))
            acc_val_locals.append(copy.deepcopy(acc_val))
        # update global weights
        # w_locals.to(args.device) # todo
        w_glob = FedAvg(args,w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss and acc
        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_train_avg = sum(acc_train_locals) / len(acc_train_locals)
        acc_val_avg = sum(acc_val_locals) / len(acc_val_locals)
        print('Round {:3d}, Average loss {:.3f} , Average acc_train {:.3f} ,Average acc_val {:.3f}'.format(iter, loss_avg,acc_train_avg,acc_val_avg))
        loss_train.append(loss_avg)
        acc_train_list.append(acc_train_avg)
        acc_val_list.append(acc_val_avg)



    # EPOCHS = 5
    # model = BertClassifier()
    # learning_rate = 1e-6
    # train(model, df_train, df_valid, learning_rate, EPOCHS)
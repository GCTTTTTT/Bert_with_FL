# !pip install transformers==4.9.2
# !pip install sentencepiece==0.1.96
from collections import defaultdict
from json import encoder

import numpy as np
import pandas as pd
import torch

# 读取BBC-news数据集
import transformers

df = pd.read_csv('bbc_news.csv')

from transformers import BertTokenizer

# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# tokenization和padding
def tokenize_articles(text_list):
    tokenized_articles = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        max_length=512,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    return tokenized_articles

# tokenized articles
tokenized_articles = tokenize_articles(df['description'].tolist())

from torch.utils.data import Dataset

class BBCNewsDataset(Dataset):
    def __init__(self, tokenized_articles, targets):
        self.tokenized_articles = tokenized_articles
        self.targets = targets

    def __getitem__(self, index):
        return {
            'input_ids': self.tokenized_articles['input_ids'][index],
            'attention_mask': self.tokenized_articles['attention_mask'][index],
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

    def __len__(self):
        return len(self.targets)

# 将数据集设置为BBCNewsDataset格式
dataset = BBCNewsDataset(tokenized_articles, df['title'].tolist())

from torch.utils.data import random_split, DataLoader

#将数据分成训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#批处理数据
batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

from transformers import BertModel


class BertClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()

        # Bert模型
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # 线性层
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 只使用Bert模型的最后一层编码
        last_hidden_state = bert_output.last_hidden_state[:, 0, :]

        # 线性层
        logits = self.linear(last_hidden_state)

        return logits

from transformers import AdamW

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertClassifier(n_classes=len(df.title.unique())).to(device)

#优化器
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

#损失函数
loss_fn = torch.nn.CrossEntropyLoss().to(device)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None, n_examples=None):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        if n_examples:
            if len(losses) * data_loader.batch_size > n_examples:
                break

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples=None):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            if n_examples:
                if len(losses) * data_loader.batch_size > n_examples:
                    break

    return correct_predictions.double() / n_examples, np.mean(losses)

# 训练模型
EPOCHS = 2
total_steps = len(train_dataloader) * EPOCHS

scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_dataset)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_dataloader,
        loss_fn,
        device,
        len(val_dataset)
    )

    print(f'Val loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

test_df = pd.read_csv('bbc_news.csv')

#对测试数据进行预测
test_df['category_pred'] = ''
model = BertClassifier(n_classes=len(test_df.title.unique())).to(device)
model.load_state_dict(torch.load('best_model_state.bin'))
model.eval()

with torch.no_grad():
    for i, row in test_df.iterrows():
        encoded_text = tokenizer.encode_plus(row['description'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)

        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

        test_df.at[i, 'category_pred'] = encoder.inverse_transform([prediction.item()])[0]

#评估模型
from sklearn.metrics import classification_report

print(classification_report(test_df['title'], test_df['category_pred']))


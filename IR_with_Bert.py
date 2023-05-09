# 使用bert-base-uncased
# 用余弦相似度评估文本相似度
# 用MAP来评估检索结果
# 代码是只测试前100条
# 加载依赖项
import math

import pandas as pd
import numpy as np
# 加载依赖项
import torch
from transformers import BertTokenizer, BertModel
# 评估函数
from sklearn.metrics import average_precision_score

np.seterr(divide='ignore',invalid='ignore')

# 加载数据集
df = pd.read_csv("bbc_news.csv", encoding='utf-8')

df = df[0:100]
# print(df)

# 将英文标题和描述组合成一列
df['text'] = df[['title','description']].apply(lambda x: ' '.join(x), axis=1)
# print(df['text'])


# from transformers

# 加载模型和tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('D:/Bert/archive/bert-base-uncased-vocab.txt')
# model = BertModel.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('D:/Bert/archive/bert-base-uncased/bert-base-uncased')
model = BertModel.from_pretrained('D:/Bert/bbcPretrainedBert')

# 建立模型函数
def get_bert_embeddings(text):
    # 对文本进行分词、添加特殊标记并编码
    encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # 将编码后的文本输入BERT模型中
    with torch.no_grad():
        model_output = model(**encoded_text)
        # 获取文本的"CLS"令牌的输出向量并返回
        embeddings = model_output.last_hidden_state[:, 0, :]
        return embeddings.numpy()

# 计算相似性函数
def get_similarity(query, text):
    query_emb = get_bert_embeddings(query)
    text_emb = get_bert_embeddings(text)

    # print(query_emb,text_emb)

    cos_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(text_emb))
    return cos_sim.item()

# 计算每个文本向量与输入字符串向量之间的余弦相似度
def rank_articles(query, df):
    return df.assign(similarity = df['text'].apply(lambda x: get_similarity(query, x)))\
             .sort_values(by='similarity', ascending=False)[['title', 'description', 'similarity']]




def evaluate(query, df):
    y_true = [1 if query in text else 0 for text in df['text'].values]
    y_scores = [get_similarity(query, text) for text in df['text'].values]
    print(y_true)
    print(y_scores)
    return average_precision_score(y_true, y_scores)

# 评估测试
query = "Ukraine, amid"
top_k = 10
results = rank_articles(query, df)[:top_k]

print(f"Top {top_k} documents with respect to query '{query}':\n")
for i, (title, desc, sim) in results.iterrows():
    print(f"{i+1}. [{sim:.4f}] {title}\n    {desc}\n")
print(f"Evalutation results for query '{query}':")
print(f"MAP: {evaluate(query, df):.4f}")



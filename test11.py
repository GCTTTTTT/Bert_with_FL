import math

import pandas as pd
import torch
from torch import cosine_similarity
from transformers import BertModel, BertTokenizer

data_path = "bbc_news.csv"    # 数据集路径
# model_name = "bert-base-uncased"    # 预训练模型名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # 设备

# 加载数据集
data = pd.read_csv(data_path)
data = data[0:100]

# 加载Bert模型和分词器
tokenizer = BertTokenizer.from_pretrained('D:/Bert/archive/bert-base-uncased-vocab.txt')
model = BertModel.from_pretrained('D:/Bert/bbcPretrainedBert').to(device)
model.eval()

# 编码文本
def encode_text(text):
    tokens = tokenizer.encode_plus(text, max_length=512, padding="max_length",
                                    truncation=True, return_tensors="pt")
    tokens.to(device)
    with torch.no_grad():
        output = model(input_ids=tokens["input_ids"],
                       attention_mask=tokens["attention_mask"])[0][:, 0, :]
    return output.cpu().numpy()

# 计算相似度
def get_similarity(input_vector, data_vectors):
    # print(input_vector,data_vectors)
    # print(input_vector,data_vectors)
    similarities = pd.DataFrame(data_vectors).apply(lambda x:
                         cosine_similarity(x.values.reshape(1, -1), input_vector), axis=1)
    return similarities[0].values
    # cos_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(input_vector), torch.from_numpy(data_vectors))
    # return cos_sim.item()

# 计算每个文本向量与输入字符串向量之间的余弦相似度
def rank_articles(query, df):
    return df.assign(similarity = df['text'].apply(lambda x: get_similarity(query, x)))\
             .sort_values(by='similarity', ascending=False)[['title', 'description', 'similarity']]


# 评估函数
def evaluate(input_str, top_k):
    # 编码输入字符串
    input_vec = encode_text(input_str)

    # 编码数据集中的description列
    data_vecs = [encode_text(text) for text in data["description"]]

    # 计算相似度
    similarities = get_similarity(input_vec, data_vecs)

    # 对结果进行排序
    top_indices = similarities.argsort()[-top_k:][::-1]

    # 计算NDCG指标
    dcg = 0
    idcg = sum([1 / math.log(i + 2, 2) for i in range(top_k)])
    for i, idx in enumerate(top_indices):
        relevance = 1 if similarities[idx] > 0.5 else 0
        dcg += relevance / math.log(i + 2, 2)
    ndcg = dcg / idcg

    # 输出结果
    for i, idx in enumerate(top_indices):
        print(f"Rank {i+1}: {data.loc[idx, 'description']}")
    print(f"NDCG@{top_k}: {ndcg}")

evaluate("technology", 5)    # 输入字符串为"technology"，输出前5个最相关的description

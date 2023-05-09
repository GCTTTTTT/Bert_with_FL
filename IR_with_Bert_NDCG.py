# 预训练Bert + cos + NDCG
import math
import pandas as pd
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

df = pd.read_csv("bbc_news.csv", encoding='utf-8')
df = df[0:100]
df['text'] = df[['title','description']].apply(lambda x: ' '.join(x), axis=1)

import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('D:/Bert/archive/bert-base-uncased-vocab.txt')
# model = BertModel.from_pretrained('D:/Bert/archive/bert-base-uncased/bert-base-uncased')
model = BertModel.from_pretrained('D:/Bert/bbcPretrainedBert')

def get_bert_embeddings(text):
    encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_text)
        embeddings = model_output.last_hidden_state[:, 0, :]
        return embeddings.numpy()

def get_similarity(query, text):
    query_emb = get_bert_embeddings(query)
    text_emb = get_bert_embeddings(text)
    cos_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(text_emb))
    return cos_sim.item()

def rank_articles(query, df):
    return df.assign(similarity = df['text'].apply(lambda x: get_similarity(query, x)))\
             .sort_values(by='similarity', ascending=False)[['title', 'description', 'similarity']]

def get_ndcg(df, k):
    """
    Computes NDCG at k for a given dataframe with columns 'relevance'
    """
    # num_relevant = np.sum(df['relevance'][:k])
    num_relevant = k # test
    # print(df['relevance'][:k])
    # print("============================")
    # print(num_relevant) # 3
    # print("============================")
    # print(np.arange(2, num_relevant+2)) # [2 3 4]
    # print("============================")
    # print(np.log2(np.arange(2, num_relevant+2)))
    # print("============================")
    ideal_dcg = np.sum(1/np.log2(np.arange(2, num_relevant+2)))
    # print(ideal_dcg)
    dcg = np.sum(df['relevance'][:k]/np.log2(np.arange(2, num_relevant+2)))
    return dcg/ideal_dcg

def evaluate(query, df, top_k):
    # Compute ground truth relevance scores
    y_true = [1 if query in text else 0 for text in df['text'].values]

    # Compute predicted relevance scores
    df = rank_articles(query, df)
    y_pred = [1 if sim > 0.955 else 0 for sim in df['similarity'].values]

    # Compute NDCG@k for predicted relevance scores
    df['relevance'] = y_pred
    ndcg = get_ndcg(df, k=top_k)

    return ndcg

# query = "Ukraine, amid"
query = "President Putin's future."
top_k = 100
results = rank_articles(query, df)

print(f"Top {top_k} documents with respect to query '{query}':\n")
for i, (title, desc, sim) in results.head(top_k).iterrows():
    print(f"{i+1}. [{sim:.4f}] {title}\n    {desc}\n")

ndcg = evaluate(query, df ,top_k)
print(f"NDCG@10: {ndcg:.4f}")
# 导入所需的库
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch

# 加载BBC-news数据集
df = pd.read_csv("bbc_news.csv")
df = df[0:100]

# 对description列数据进行预处理
df["description"] = df["description"].str.lower()  # 转换为小写字母
df["description"] = df["description"].str.replace(r'[^\w\s]', '')  # 去除特殊字符

# 训练BM25模型计算文本相关性
corpus = df["description"].tolist()
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 加载预训练的Bert模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 定义计算相似度函数
def calculate_similarity(input_text, bm25, corpus, model, tokenizer):
    # 对输入文本进行预处理和分词
    input_text = input_text.lower().replace(r'[^\w\s]', '')
    input_tokens = tokenizer.encode(input_text, add_special_tokens=False)

    # 获取输入文本的Bert嵌入向量表示
    input_ids = torch.tensor([input_tokens])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
        input_embedding = last_hidden_states.mean(dim=1).squeeze()

    # 计算输入文本和所有description的BM25相似度
    scores = bm25.get_scores(input_tokens)
    # 将BM25相似度乘以Bert嵌入向量相似度，得到最终的相似度得分
    similarities = scores * input_embedding.cosine_similarity(model(tokenizer(corpus, padding=True, truncation=True, return_tensors='pt'))[0]).flatten()

    return similarities

# 定义执行信息检索任务的函数
def information_retrieval(query, df, bm25, tokenizer, model):
    # 计算相似度得分
    similarities = calculate_similarity(query, bm25, df["description"].tolist(), model, tokenizer)
    # 获取得分前n个文本条目
    n = 10
    top_n_indices = similarities.argsort()[-n:][::-1]
    top_n_results = df.iloc[top_n_indices][["title", "description"]]

    return top_n_results

# 执行信息检索任务
query = "China"
results = information_retrieval(query, df, bm25, tokenizer, model)
print(results)
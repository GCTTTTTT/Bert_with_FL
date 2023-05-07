import pandas as pd

# 读取BBC-news数据集
df = pd.read_csv("../../bbc_news.csv")
# print(df['title'])
l1 = []
l2 = []
cnt = 0
for line in df['title']:
    l1.append(line)

for line in df['description']:
    l2.append(line)
# cnt=0
f = open("test1.txt", 'w+', encoding='utf8')
for i in range(len(l1)):
    s = l1[i] + " " + l2[i] + '\n'
    f.write(s)
    # cnt+=1
    # if cnt>10: break
f.close()
# print(l1)


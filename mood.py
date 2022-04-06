# coding = utf-8
import tushare as ts
import pandas as pd
import jieba
import codecs
import numpy as np
import matplotlib.pyplot as plt


# %% 创建一个情感词典
def dict_initial(dictname='./Chinese_document.xlsx', nofile="否定.txt"):
    #     @param:dictname,姜老师的中文情感词典文件路径
    #     @param:nofile,否定词文件的路径
    #     @return:moodworddict-情感词典，其中消极词汇对应-1，积极词汇对应1
    #     @return:noworddict-否定词词典，为了提高查找效率做成了dict的形式
    # 获取消极词汇并更新词典
    moodworddict = {}
    neg = pd.read_excel(dictname, sheet_name="negative")
    for value in neg.values:
        value = value[0].strip()
        moodworddict.update({value: -1})
    # 获取积极词汇并更新词典
    pos = pd.read_excel(dictname, sheet_name="positive")
    for value in pos.values:
        value = value[0].strip()
        moodworddict.update({value: 1})
    noworddict = {}
    # 获取否定词，将之放入否定词典中
    with open(nofile, encoding='gbk') as nowordfile:
        noword = nowordfile.read()
        nowordlist = noword.split('\n')
        nowordlist.remove('')
        for word in nowordlist:
            noworddict.update({word: 0})
    #     print(moodworddict)
    #     print(noworddict)
    return moodworddict, noworddict


# %% 获取数据，返回df
def get_data(start, end):
    ts.set_token('8015a7efefba12b33fc0581ab86ec2ff5ce1827526924b9ab13ef102')
    pro = ts.pro_api()
    df = pro.news(src='sina', start_date=start, end_date=end)
    df = pd.DataFrame(df)
    new_name = ['datetime', 'content', 'marks']
    df.columns = new_name
    return df


# %% 数据预处理
def process_data(df):
    length = df.shape[0]
    stopwords = [line.strip() for line in codecs.open('中文停用词表.txt', 'r', 'utf-8').readlines()]
    lists = [[] for i in range(length)]
    for i in range(length):
        # print(i)
        content = df.iat[i, 1]
        ls = jieba.lcut(content, cut_all=False)
        # print(ls)
        for word in ls:
            if word not in stopwords:
                lists[i].append(word)
    # 去掉文本中的空格
    for i in range(len(lists)):
        for j in range(lists[i].count(' ')):
            lists[i].remove(' ')
    return lists


# %% 获得情感分数
def get_text_sentiment(text, moodworddict, noworddict):
    #     @param:text,要分析的文章
    #     @param:moodsworddict,情感词典
    #     @param:noworddict,否定词词典
    #     返回值为分析后的情感结果
    moodwordcount = 0
    nocount = 0
    sentiment = 0
    for word in text:
        if (word in moodworddict):  # 找到一个情绪词
            sentiment += np.power(-1, nocount) * moodworddict.get(word)
            moodwordcount += 1
            nocount = 0
        elif (word in noworddict):
            nocount += 1
    if moodwordcount == 0:
        sentiment = 0
    else:
        sentiment /= moodwordcount
    return sentiment


# %%
def mood(start, end, moodworddict, noworddict):
    df = get_data(start, end)
    lists = process_data(df)
    for i in range(len(lists)):
        mark = get_text_sentiment(lists[i], moodworddict, noworddict)
        df.iloc[i][2] = mark
    return df


# %% 由于有单次1000条限制，所以需要生成一个时间序列，一日日调用
def timeseries(start, end):
    t = pd.date_range(start=start, end=end)
    a = pd.DataFrame()
    a = pd.Series(index=t)
    return a


# %% 创建一个新的列表用于存储
df2 = timeseries('20210101', '20220101')
df2 = df2.reset_index()
df2['total_mark'] = 0.0
df2['count'] = 0
df2['average'] = 0.0
df2 = df2.drop(labels=0, axis=1)  # 不知道为啥有个0列，貌似上面没创建好
df2['index'] = pd.to_datetime(df2['index'])  # 对时间进行一次处理，只保留日期
df2['index'] = df2['index'].dt.date
df2['index'] = df2['index'].astype("string")  # 转回字符串，而不是datetime格式

# %%
moodworddict, noworddict = dict_initial()
for i in range(len(df2) - 1):
    start = df2.iloc[i][0]
    end = df2.iloc[i + 1][0]
    df = mood(start, end, moodworddict, noworddict)
    total = 0
    count = 0
    for j in range(len(df)):
        if df.iloc[j][2] != 0:  # 去掉非零的
            total += df.iloc[j][2]  # 该日总分
            count += 1  # 计数
    df2.iat[i, 1] = total
    df2.iat[i, 2] = count
    df2.iat[i, 3] = total / count

# %% 保存以及画图
# df2.to_csv('sina2021.csv')
df2.plot(x='index', y='average')
plt.show()

import codecs
import numpy as np
import jieba
k = 6
stop_word = []
def read_stop_word(stop_word):
    res = []
    with codecs.open(stop_word,'rb','utf-8') as csvfile:
        for row in csvfile:
            res.append(row.split('\n')[0])
    return res
def read_data(train,test_labled):
    train_res = []
    train_label = []
    test_data = []
    test_label = [] 
    countOne = 0
    with codecs.open(train,'rb','utf-8') as csvfile:
        csvfile.readline();
        for row in csvfile:
            line = row.rsplit(',',1)
            if int(line[1]) == -1:
                countOne += 1
            train_res.append(line[0])
            train_label.append(line[1].split('\r')[0])
    with codecs.open(test_labled,'rb','utf-8') as csvfile:
        csvfile.readline();
        for row in csvfile:
            line = row.rsplit(',',1)
            if int(line[1]) == -1:
                countOne += 1
            test_data.append(line[0])
            test_label.append(line[1])
    return np.array(train_res),np.array(train_label),np.array(test_data),np.array(test_label)


def train(test_data,test_label,train_data,train_label):
    for i in range(1):
        label = test(test_data[i],train_data,train_label)
        print(label,test_label[i])
def test(test_data,train_data,train_label):
    seg_list = jieba.cut(test_data,cut_all=False)
    count = np.zeros(len(train_data))
    for i in range(len(train_data)):
        for w in seg_list:
            if w not in stop_word and w in train_data[i]:
               count[i] += 1
    sorted_count = count.argsort()
    class_count = {}
    for i in range(k):
        vote_label = train_label[sorted_count[i]]
        class_count[vote_label] = class_count.get(vote_label,0) + 1
    sorted_vote_count = sorted(class_count.items(),key=lambda item:item[1],reverse = True)
    print(sorted_vote_count)
    return sorted_vote_count[0][0]
stop_word = read_stop_word("../stopwords-master/cn_stopwords.txt")
print(stop_word)
train_data,train_label,test_data,test_label = read_data("../data/train.csv","../data/test_labled.csv")
train(test_data,test_label,train_data,train_label)
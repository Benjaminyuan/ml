import csv
import math
import codecs
import numpy as np
import jieba
C = [-1,0,1]
vocabulary = set()
def read_stop_word(stop_word):
    res = []
    with codecs.open(stop_word,'rb','utf-8') as csvfile:
        for row in csvfile:
            res.append(row.split('\n')[0])
    return res
def train(docu,C):
    docu_class_map = {}
    logprior = {}
    N_doc = docu.shape[0]
    bigdoc = {}
    loglikelihood = {}
    cset = set()
    for a in docu:
        t = docu_class_map.get(int(a[1]),[])
        if t is None:
            t = []
        t.append(a[0])
        docu_class_map[int(a[1])] = t
    # print(docu_class_map)
    # for a in docu:
    #     seg_list = jieba.cut(a[0],cut_all=False)
    #     vocabulary.update(seg_list)
    #遍历每一种类别
    for c in C:
        #获取类别列表
        N_clist = docu_class_map[int(c)]
        logprior[c] = math.log(len(N_clist)/N_doc,2)
        #count用于统计c类评论中，单词出现的次数
        count = {}
        # 读取评论
        for line in N_clist:
            # 评论分词
            seg_list = jieba.cut(line,cut_all=False)
            # 统计
            for w in seg_list:
                count[w] = count.get(w,0)+1
        # 对全局的词进行统计
        wsum = 0;
        for i in count.values():
            wsum += i
        for w in vocabulary:
            wcout = loglikelihood.get(w,{})
            wcout[c] = math.log((count.get(w,0)+1)/(wsum+len(vocabulary)),2)
            loglikelihood[w] = wcout;
    return logprior,loglikelihood

def read_data(train,test_labled):
    train_res = []
    test_data = []
    test_label = [] 
    countOne = 0
    with codecs.open(train,'rb','utf-8') as csvfile:
        csvfile.readline();
        for row in csvfile:
            line = row.rsplit(',',1)
            if int(line[1]) == -1:
                countOne += 1
            train_res.append([line[0],line[1].split('\r')[0]])
    with codecs.open(test_labled,'rb','utf-8') as csvfile:
        csvfile.readline();
        for row in csvfile:
            line = row.rsplit(',',1)
            if int(line[1]) == -1:
                countOne += 1
            test_data.append(line[0])
            test_label.append(line[1])
    return np.array(train_res),np.array(test_data),np.array(test_label)
def test_naive_bayes(test_doc,logprior,loglikelihood,C):
    sum = {}
    for c in C:
        sum[c] = logprior[c]
        # 分词
        seg_list = jieba.cut(test_doc,cut_all=False)
        for w in seg_list:
            if w in vocabulary:
                sum[c] += sum[c] + loglikelihood[w][c]
    maxC = '0'
    maxN = sum[maxC]
    for c in C:
        if sum[c] > maxN:
            maxN = sum[c]
            maxC = c
    return maxN,maxC
def test(test_list,test_label,logprior,loglikelihood,C):
    res = []
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    R = 0
    for i in range(len(test_list)):
        maxN,maxC = test_naive_bayes(test_list[i],logprior,loglikelihood,C)
        print(maxC,test_label[i])
        if int(maxC) == int(test_label[i]):
            R += 1 
            if maxC == '1':
                TP += 1
            elif maxC == '-1':
                TN += 1
        else:
            if int(maxC) == 1 and int(test_label[i]) == -1:
                FP += 1 
            elif int(maxC) == -1 and int(test_label[i]) == 1:
                FN += 1 
    print(R,TP,TN,FP,FN)
    precision = TP/(TP+FP)
    print('pricision: ',precision)
    recall = TP/(TP+FN)
    print('recall',recall)
    print('F1-score:',2*precision*recall/(precision+recall))
def countWord(train_data):
    C = ['-1','0','1']
    word_count = {}
    stop_words = read_stop_word("../stopwords-master/cn_stopwords.txt")
    for line in train_data:
        seg_list = jieba.cut(line[0],cut_all=False)
        c = line[1]
        for w in seg_list:
            c_word_count = word_count.get(w,{})
            c_word_count[c] = c_word_count.get(c,0)+1
            word_count[w] = c_word_count
    c_count_list = {}
    for c in C:
        m = {}
        for w in word_count.keys():
            wc_map = word_count[w]
            if c in wc_map:
                m[w] = wc_map.get(c,0)
        c_count_list[c] = m
    c_word_sorted_count = sorted(c_count_list['-1'].items(),key=lambda item:item[1],reverse = True)
    c_word_sorted_count.extend(sorted(c_count_list['0'].items(),key=lambda item:item[1],reverse = True))
    c_word_sorted_count.extend(sorted(c_count_list['1'].items(),key=lambda item:item[1],reverse = True))
    for word in c_word_sorted_count:
        if word[0] not in stop_words and word[1] > 10:
            print(word)
            vocabulary.add(word[0])
    # print(vocabulary)
    return 

train_data,test_data,test_label = read_data("../data/train.csv","../data/test_labled.csv")

c = ['-1','0','1']
# print(train_data)
countWord(train_data)
logprior,loglikelihood = train(train_data,c)
print(logprior)
test(test_data,test_label,logprior,loglikelihood,c)
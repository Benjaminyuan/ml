import csv
import math
import codecs
import numpy as np
import jieba
C = [-1,0,1]
def train(docu,C):
    docu_class_map = {}
    vocabulary = set()
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
    for a in docu:
        seg_list = jieba.cut(a[0],cut_all=False)
        vocabulary.update(seg_list)
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
        for w in vocabulary:
            wcout = loglikelihood.get(w,{})
            wcout[c] = wcout.get(c,0)+count.get(w,0)
            loglikelihood[w] = wcout;
    return logprior,loglikelihood
def read_data(test_label):
    res = []
    countOne = 0
    with codecs.open(test_label,'rb','utf-8') as csvfile:
        # reader = csv.DictReader(csvfile)
        # for row in reader:
        #     print(row)
        csvfile.readline();
        for row in csvfile:
            line = row.rsplit(',',1)
            if int(line[1]) == -1:
                countOne += 1
            res.append([line[0],line[1]])
    print(countOne)
    return np.array(res)
res = read_data("../data/test_labled.csv")
c = ['-1','0','1']
logprior,loglikelihood = train(res,c)
print(logprior)
print(loglikelihood)
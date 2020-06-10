import codecs
import numpy as np
import jieba
k = 5
stop_word = []
# 词空间
vocabulary = set()
words_idx = {}
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
            test_label.append(line[1].split('\r')[0])
    return np.array(train_res),np.array(train_label),np.array(test_data),np.array(test_label)


def train(test_data,test_label,train_data,train_label):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    R = 0
    for i in range(200):
        label = classify0(test_data[i],train_data,train_label)
        print(label,test_label[i])
        if int(label) == int(test_label[i]):
            R += 1 
            if int(label) == 1:
                TP += 1
            elif int(label) == -1:
                TN += 1
        else:
            if int(label) == 1 and int(test_label[i]) == -1:
                FP += 1 
            elif int(label) == -1 and int(test_label[i]) == 1:
                FN += 1 
    print(R,TP,TN,FP,FN)
    precision = TP/(TP+FP)
    print('pricision: ',precision)
    recall = TP/(TP+FN)
    print('recall',recall)
    print('F1-score:',2*precision*recall/(precision+recall))
def classify0(inX,data_set,labels):
    data_set_size = data_set.shape[0]
    # 求和每个训练数据的距离差
    diff_matrix =  np.tile(inX,(data_set_size,1)) - data_set
    # 求和每个训练数据的距离差平方、求和、开方
    sqrt_diff_matrix = diff_matrix**2
    sqrt_distance = sqrt_diff_matrix.sum(axis=1)
    sqrt_distance = sqrt_distance ** 0.5
    sorted_distance_indicies = sqrt_distance.argsort()
    class_count = {}
    for i in range(k):
        #投票
        vote_label = labels[sorted_distance_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label,0) + 1
    sorted_vote_count = sorted(class_count.items(),key=lambda item:item[1],reverse = True)
    return sorted_vote_count[0][0]
def countWord(train_data,train_label):
    C = ['-1','0','1']
    word_count = {}
    stop_words = read_stop_word("../stopwords-master/cn_stopwords.txt")
    for i in range(len(train_data)):
        seg_list = jieba.cut(train_data[i],cut_all=False)
        c = train_label[i]
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
    print(len(c_word_sorted_count))
    l0 = sorted(c_count_list['0'].items(),key=lambda item:item[1],reverse = True)
    c_word_sorted_count.extend(l0)
    l1 = sorted(c_count_list['1'].items(),key=lambda item:item[1],reverse = True)
    c_word_sorted_count.extend(l1)
    for word in c_word_sorted_count:
        if word[0] not in stop_words and word[1] > 150:
            vocabulary.add(word[0])
    i = 0
    for w in vocabulary:
        words_idx[w] = i
        i += 1 
    # print(len(l0),len(l1),len(vocabulary))
    return 
def word_to_vec(words,vocabulary):
    words = list(words)
    size = len(vocabulary)
    res = np.zeros(size)
    words_set = set(words)
    for w in words_set:
        if w in vocabulary:
            # print(words_idx[w],words.count(w))
            res[words_idx[w]] = words.count(w)
    return res
def data_vec(data):
    res = []
    for w in data:
        seg_list = list(jieba.cut(w,cut_all=False))
        res.append(word_to_vec(seg_list,vocabulary))
    # print(np.array(res))
    return np.array(res)
    
stop_word = read_stop_word("../stopwords-master/cn_stopwords.txt")
train_data,train_label,test_data,test_label = read_data("../data/train.csv","../data/test_labled.csv")
countWord(train_data,train_label)
train_data = data_vec(train_data)
test_data = data_vec(test_data)
train(test_data,test_label,train_data,train_label)
import codecs
import numpy as np
import jieba
k = 5
stop_word = []
vocabulary_naga = set()
vocabulary_zero = set()
vocabulary_pos = set()
def Knn_classfy(newInput, newmarkedwords, dataset,datasetmarkedwords, labels,k):
    squaredDist=[0 for j in range(len(dataset))]
    distance=[0 for j in range(len(dataset))]
    for i in range(len(dataset)):
        for j in range(len(datasetmarkedwords[i])):
            column=datasetmarkedwords[i][j]
            squaredDist[i] +=(dataset[i][column]-newInput[column]) ** 2
        for j in range(len(newmarkedwords)):
            column=newmarkedwords[j]
            squaredDist[i] +=(dataset[i][column]-newInput[column]) ** 2
        distance[i]=squaredDist[i] ** 0.5
        
    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = np.argsort(distance)
	    
    positive_label=0
    negative_label=0
    predict=0
    for i in range(k):  
        ## step 3: choose the min k distance  
	    voteLabel = labels[sortedDistIndices[i]]
	    ## step 4: count the times labels occur  
	    if(voteLabel==1):
	        positive_label+=1
	    else :
	        negative_label+=1
    ## step 5: the max voted class will return
    if(positive_label>=negative_label):
        predict=1
    else:
        predict=0
    return predict

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
    for i in range(len(test_data)):
        label = test(test_data[i],train_data,train_label)
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
def test(test_data,train_data,train_label):
    seg_list = jieba.cut(test_data,cut_all=False)
    count = np.zeros(len(train_data))
    for i in range(len(train_data)):
        for w in seg_list:
            if w in vocabulary_naga and w in train_data[i]:
               count[i] += 6
            elif w in vocabulary_pos and w in train_data[i]:
               count[i] += 1
            elif w in vocabulary_zero and w in train_data[i]:
               count[i] += 2
           
    sorted_count = count.argsort()
    class_count = {}
    for i in range(k):
        vote_label = train_label[sorted_count[i]]
        class_count[vote_label] = class_count.get(vote_label,0) + 1
    sorted_vote_count = sorted(class_count.items(),key=lambda item:item[1],reverse = True)
    print(sorted_vote_count)
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
    word_0_sorted_count = sorted(c_count_list['0'].items(),key=lambda item:item[1],reverse = True)
    word_1_sorted_count = sorted(c_count_list['1'].items(),key=lambda item:item[1],reverse = True)
    for word in c_word_sorted_count:
        if word[0] not in stop_words and word[1] > 100:
            print(word)
            vocabulary_naga.add(word[0])
    for word in word_0_sorted_count:
        if word[0] not in stop_words and word[1] > 1000:
            print(word)
            vocabulary_zero.add(word[0])
    for word in word_1_sorted_count:
        if word[0] not in stop_words and word[1] > 1000:
            print(word)
            vocabulary_pos.add(word[0])
    # print(vocabulary)
    return 
stop_word = read_stop_word("../stopwords-master/cn_stopwords.txt")
print(stop_word)
train_data,train_label,test_data,test_label = read_data("../data/train.csv","../data/test_labled.csv")
countWord(train_data,train_label)
train(test_data,test_label,train_data,train_label)
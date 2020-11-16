import re
import jieba
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.metrics import  classification_report
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

#coding:utf-8
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#设置最频繁使用的50000个词
MAX_NB_WORDS=50000
#每条分词后的微博最大的长度
MAX_SEQUENCE_LENGTH=250
#设置embedding层的维度
EMBEDDING_DIM=100


#用于删除 除了字母、数字、汉字以外的符号
def remove_punctuation(line):
    line=str(line)
    if line.strip()=='':
        return ''
    rule=re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line=rule.sub('',line)
    return line


def readData(filename):
    df = pd.read_csv(filename)
    df['content_after_remove'] = df['微博中文内容'].apply(remove_punctuation)
    df['content_after_cut_filter'] = df['content_after_remove'].apply(
        lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
    # df['labelChinese']= df['情感倾向'].apply(intToLabel)
    return df

def modulePrework(df):
    X = tokenizer.texts_to_sequences(df['content_after_cut_filter'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = pd.get_dummies(df['情感倾向']).values
    return X,Y


if __name__=='__main__':
    #读取中文停等词
    stopwords = [line.strip() for line in open("../stopwords-master/cn_stopwords.txt", 'r', encoding='utf-8').readlines()]
    #处理训练集数据
    train=readData("../data/train.csv")
    #处理测试集数据
    test=readData("../data/test_labled.csv")

    df = pd.read_csv('data/test_labled.csv')
    id_df=df[['情感倾向']].drop_duplicates().sort_values('情感倾向')
    print(id_df)
    print(id_df['情感倾向'].values)
    print(id_df[['情感倾向']].values)

    # LSTM建模开始
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(train['content_after_cut_filter'].values)

    #准备建模初始
    X_train,Y_train = modulePrework(train)
    X_test,Y_test=modulePrework(test)

    #定义模型
    model=Sequential()
    model.add(Embedding(MAX_NB_WORDS,EMBEDDING_DIM,input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    print("--------------------------------------")
    #训练数据
    epochs=5
    batch_size=64
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    #画损失函数趋势图和准确率趋势图
    plt.title("损失函数趋势图")
    plt.plot(history.history['loss'],label='tarin')
    plt.plot(history.history['val_loss'],label='verify')
    plt.legend()
    plt.show()
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'],label='train')
    plt.plot(history.history['val_accuracy'],label='verify')
    plt.legend()
    plt.show()

    #模型分析
    y_pred=model.predict(X_test)
    y_pred=y_pred.argmax(axis=1)
    Y_test=Y_test.argmax(axis=1)


    #三个分析数据的表
    print('accuracy:%s'%accuracy_score(y_pred,Y_test))
    print(classification_report(Y_test,y_pred))

    # 生成混淆矩阵
    conf_mat = confusion_matrix(Y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=id_df['情感倾向'].values, yticklabels=id_df['情感倾向'].values)
    plt.xlabel('预测结果', fontsize=18)
    plt.ylabel('实际结果', fontsize=18)
    plt.show()
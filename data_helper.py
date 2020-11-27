import os
import numpy as np
import re
import gensim

"""
载入句子
输入：
#   file_dir  文件夹路径
#   file_names 文件夹下文件名称列表
#返回值：
#   all_sentences 所有句子组成的列表
#   all_labels  所有句子对应的标签，在大于5分的评论里则为正标签
"""
def load_data_and_labels_1(file_dir, file_names):
    all_sentences = []
    all_labels = []
    sentences = []
    for file in file_names:
        name = file
        file = os.path.join(file_dir, file)
        with open(file) as f:
            document = list(f)
            #sentences = [re.sub(r'"{2,}', "", line).split() for line in sentences]
            for line in document:
                line = re.sub(r'"{2,}', "", line)
                line = re.sub(r"^\d.\b", "", line)
                #sentences.append(line.lower().split())
                sentences.append(line.lower().strip())
            all_sentences.extend(sentences)
            if "neg" in name:
                labels = [0,1] * len(sentences)
            else:
                labels = [1,0] * len(sentences)
            all_labels.extend(labels)
    all_labels = np.array(all_labels)
    return all_sentences, all_labels


def load_data_and_labels(file_dir, file_names):
    pos_sentences = []
    neg_sentences=[]
    neg_labels = []
    pos_labels =[]
    for file in file_names:
        name = file
        file = os.path.join(file_dir, file)
        with open(file) as f:
            document = list(f)
            if "neg" in name:
                for line in document:
                    line = preprocess_sentence(line)
                    neg_sentences.append(line)
                    neg_labels.append([0,1])
            else:
                for line in document:
                    line = preprocess_sentence(line)
                    pos_sentences.append(line)
                    pos_labels.append([1, 0])
    neg_labels = np.array(neg_labels)
    pos_labels = np.array(pos_labels)
    return neg_sentences, neg_labels, pos_sentences, pos_labels

def preprocess_sentence(sentence):
    sentence = re.sub(r'"{2,}', "", sentence)
    sentence = re.sub(r"^\d.\b", "", sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'\d+','', sentence)
    sentence = re.sub(r'[,\.!@\*()]', '', sentence)
    sentence = sentence.strip()
    return  sentence





"""
获取词嵌入层的参数W
输入：
    vocab_ids_map {词汇:索引}
    fname  word2vec二进制文件
    vocab  需要的词汇
    ksize  嵌入维度
输出：
    矩阵W，W[i]代表下标为i的词的词向量
    当词汇不在词汇表出现时候，用随机初始化
"""

def get_W(vocab, fname, voacb_ids_map, ksize = 300):
    print("开始构造嵌入矩阵W...")
    W = np.zeros(shape=[len(vocab), ksize]).astype(np.float32)
    model = gensim.models.KeyedVectors.load_word2vec_format(fname,binary=True)
    for word in vocab:
        try:
            W[voacb_ids_map[word]] = model[word]
        except:
            W[voacb_ids_map[word]] = np.random.uniform(-1,1,size=ksize).astype(np.float32)
    return W

"""
获取batch的迭代器
输入：
    data 总的数据
    batch_size 

"""
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def embedding_batch(x_pos, x_neg):
    x = np.random.randint(0,2)
    pos_len = len(x_pos)
    neg_len = len(x_neg)
    if x== 0:
        index_a = np.random.randint(0, pos_len)
        index_p = np.random.randint(0, pos_len)
        index_n = np.random.randint(0, neg_len)
        anchor = x_pos[index_a]
        positive = x_pos[index_p]
        negative = x_neg[index_n]
    else:
        index_a = np.random.randint(0,neg_len)
        index_p = np.random.randint(0, neg_len)
        index_n = np.random.randint(0, pos_len)
        anchor = x_neg[index_a]
        positive = x_neg[index_p]
        negative = x_pos[index_n]

    return np.array([anchor, positive, negative])

def  preprocess_classify_sentence(sentence):
    if(re.match(r'positive',sentence)):
        label = [1,0]
    else:
        label = [0,1]
    sentence = re.sub(r'^\w*,\w*,', '', sentence)
    sentence = re.sub(r'\W', ' ', sentence)
    sentence = sentence.strip()
    return sentence.lower(), label

def load_classify_data(file_name):
    sentences = []
    labels = []
    with open(file_name) as f:
        document = list(f)
        for sentence in document:
            sentence,label = preprocess_classify_sentence(sentence)
            sentences.append(sentence)
            labels.append(label)
    return sentences, np.array(labels)

def max_sentence_length(documents):
    max_len = 0
    for document in documents:
        max_len=max(max_len, max( [len(sentence.split(' ')) for sentence in document] ))
    return max_len
def classify_batch(x,y,batch_size):
    index = np.random.randint(0,len(x),size=(batch_size))
    return x[index], y[index]
def classify_batch_iter(x,y,batch_size):
    for i in range(0,len(x),batch_size):
        yield (x[i:i+batch_size], y[i:i+batch_size])

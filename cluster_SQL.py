import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime
#Remember 0: update, 1: get, 2: delete, 3: insert, 4: misc
def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d') #to check the if the format matches SQL date format
        return True
    except ValueError:
        return False


split= [] #contains a list 
num_of_types_of_queries= {}

def splitter(file):
    update = [] #initialise empty lists for the main 5 types of queries
    get = [] #read
    delete = []
    insert = []
    misc = []
    # file = pd.to_frame(file)
    queries = file['QUERY']
    queries = queries.to_frame()
    for i in range(0, len(queries)):
        queries.iat[i, 0]= queries.iat[i, 0].lower()
        if((queries.iat[i, 0])[slice(6)]=='select'):
            get.append(queries.iat[i, 0])
        elif((queries.iat[i, 0])[slice(6)]=='update'):
            update.append(queries.iat[i, 0])
        elif((queries.iat[i, 0])[slice(6)]=='delete'):
            delete.append(queries.iat[i, 0])
        elif((queries.iat[i, 0])[slice(6)]=='insert'):
            insert.append(queries.iat[i, 0])
        else:
            misc.append(queries.iat[i, 0]) 
    split.append(update) #0 is update
    split.append(get) #1 is get
    split.append(delete) #2 is delete 
    split.append(insert) #3 is insert
    split.append(misc) #4 is misc
    return split

def num_queries(split):
    for i in range(0, len(split)):
        num_of_types_of_queries[i]= len(split[i])
    return num_of_types_of_queries #return the count of the queries of each type

def gen_module(get):
    # print(get)
    final_api =  pd.DataFrame(columns=['text', 'datatype']) #contain all the APIs irrespective of the type of query
    tokenizer = Tokenizer(
    num_words=5000,
    filters='!"#$%&+-/:;?@[\\]^{|}~\t\n', #these will be ignored during tokenization
    lower=True, #makes all the queries in lowercase
    split=' ')
    tokenizer.fit_on_texts(get)
    # print(tokenizer)
    word_index = tokenizer.word_index
    training_sequences = tokenizer.texts_to_sequences(get)
    # print(training_sequences)
    training_padded = pad_sequences(training_sequences,maxlen=100, 
                                truncating= 'post', padding='post') #all sequences are made of equal length
    # t_sne= TSNE(n_components=3, perplexity= 30, learning_rate='auto', 
    #             init='random', n_iter= 5000) #dimension reduction using t_SNE
    # train_embedded= t_sne.fit_transform(training_padded) 
    # print(training_padded) 
    pca= PCA(n_components= 3)
    train_embedded= pca.fit_transform(training_padded)
    print(train_embedded.shape)
    scaler= MinMaxScaler()
    train_scaled= scaler.fit_transform(train_embedded)
    model63 = DBSCAN(eps=0.03,
               min_samples=10,
               metric='euclidean',
               metric_params=None,
               algorithm='auto',
               leaf_size=30,
               p=None,
               n_jobs=None, 
              )
    clm63= model63.fit(train_scaled)
    print(clm63.labels_)
    get= pd.DataFrame(get)
    get['class']= clm63.labels_ #adds class to each query based upon the cluster
    get= get.sort_values(by=['class'])
    get_class={} #initialize a dictionary to store the APIs
    print("Hello")
    final_command= pd.DataFrame(columns=['text', 'datatype'])
    print(final_command)
    print(max(clm63.labels_)+1)
    for i in range(0, max(clm63.labels_)+1):
        get_class[i]= get.loc[get['class']==i]
    print("hello1")
    for i in range(0, max(clm63.labels_)+1):
        print("printhello")
        fin_seq = tokenizer.texts_to_sequences(get_class[i].loc[:,0])
        fin_padded = pad_sequences(fin_seq,maxlen=100, 
                                   truncating= 'post', padding='post')
        final= fin_padded[0]*len(fin_padded)
        for j in range(1, len(get_class[i])):
                final= fin_padded[0]-fin_padded[j]
        conflicts=[]
        datatype=[]
        txt= get_class[i].iat[0,0]
        x= txt.split()
        for k in range(0, len(final)):
            if(final[k]!=0 and k<len(x)):
                conflicts.append(k) #notes the conflicts locations
                if(x[k].isnumeric()):
                    datatype.append('int')
                elif(validate_date(x[k])):
                    datatype.append('date')
                else:
                    datatype.append('str')
                x[k]= '{}' #replaces conflict points with placeholder
        final_command.loc[i, 'text']= (" ".join(x))
        final_command.loc[i, 'datatype']= datatype
    final_api= final_command.drop_duplicates(subset=['text'])
    print(final_api)
    return final_api


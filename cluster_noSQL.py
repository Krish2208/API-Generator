import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime
import re
#Remember 0: update, 1: get, 2: delete, 3: insert_one, 4: insertMany 5: misc

def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

split= []
final_api= []


def splitter(path):
    insert_one= []
    insert_many= []
    get= []
    update= []
    delete= []
    misc= []
    queries= pd.read_csv('')
    for i in range(0, len(queries)):
        queries.iat[i, 0]= queries.iat[i,0].lower()
        txt= queries.iat[i, 0]
        x= re.split(r'[.()]', txt)
        if('insert' in x):
          insert_one.append(txt)
        elif('insertMany' in x):
          insert_many.append(txt)
        elif('find' in x):
          get.append(txt)
        elif('update' in x):
          update.append(txt)
        elif('remove' in x):
          delete.append(txt)
        else:
          misc.append(txt)
    split.append(update) #0 is update
    split.append(get) #1 is get
    split.append(delete) #2 is delete 
    split.append(insert_one) #3 is insert_one
    split.append(insert_many) #4 is insertMany
    split.append(misc) #5 is misc
    return split


def gen_module(get):
    tokenizer = Tokenizer(
    num_words=5000,
    filters='!"#$%&()+-./:;<>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ')
    tokenizer.fit_on_texts(get)
    word_index = tokenizer.word_index
    training_sequences = tokenizer.texts_to_sequences(get)
    training_padded = pad_sequences(training_sequences,maxlen=40, 
                                truncating= 'post', padding='post')
    t_sne= TSNE(n_components=3, perplexity= 30, learning_rate='auto', 
                init='random', n_iter= 5000)
    train_embedded= t_sne.fit_transform(training_padded)  
    scaler= MinMaxScaler()
    train_scaled= scaler.fit_transform(train_embedded)
    model63 = DBSCAN(eps=0.06,
               min_samples=2,
               metric='euclidean',
               metric_params=None,
               algorithm='auto',
               leaf_size=30,
               p=None,
               n_jobs=None, 
              )
    clm63= model63.fit(train_scaled)
    get= pd.DataFrame(get)
    get['class']= clm63.labels_
    get= get.sort_values(by=['class'])
    get_class={}
    for i in range(0, max(clm63.labels_)+1):
        get_class[i]= get.loc[get['class']==i]
    get_final_commands= []
    for i in range(0, max(clm63.labels_)+1):
        fin_seq = tokenizer.texts_to_sequences(get_class[i].loc[:,0])
        fin_padded = pad_sequences(fin_seq,maxlen=40, 
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
                conflicts.append(k)
                x[k]= '{}'
                if(x[k].isnumeric()==True):
                    datatype.append('int')
                elif(validate_date(x[k])==True):
                    datatype.append('date')
                else:
                    datatype.append('str')
        k={}
        k['text']= (" ".join(x))
        k['datatype']= datatype
        get_final_commands.append(k)
    get_final_commands= pd.DataFrame(get_final_commands)
    final_api.append(get_final_commands)
    return final_api

#example
path= '/content/Queries_compile.csv'
split= splitter(path)
for i in range(0, len(split)):
  try:
    final_api= gen_module(split[i])
  except ValueError:
    print('No values')
final_api= pd.DataFrame(final_api)
final_api.to_csv('final.csv')
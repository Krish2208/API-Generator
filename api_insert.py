import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import re
from datetime import datetime
# Assuming splitter has been run and the list with insert queries is available

def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# Pass the list with insert queries to the generator_insert function
def generator_insert(get):
    final_api = pd.DataFrame(columns=['text', 'datatype'])
    getx = get.copy()
    for i in range(len(getx)):
        getx[i] = getx[i].partition('VALUES')[0]
    print(get)
    tokenizer = Tokenizer(
        num_words=5000,
        filters='!"#$%&+-/:;?@[\\]^{|}~\t\n',
        lower=True,
        split=' ')
    tokenizer.fit_on_texts(getx)
    word_index = tokenizer.word_index
    training_sequences = tokenizer.texts_to_sequences(getx)
    training_padded = pad_sequences(training_sequences, maxlen=40,
                                    truncating='post', padding='post')
    pca = PCA(n_components=3)
    train_embedded = pca.fit_transform(training_padded)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_embedded)
    model63 = DBSCAN(eps=0.03,
                     min_samples=30,  # 0.04 and 10
                     metric='euclidean',
                     metric_params=None,
                     algorithm='auto',
                     leaf_size=30,
                     p=None,
                     n_jobs=None,
                     )
    clm63 = model63.fit(train_scaled)
    get = pd.DataFrame(get, columns=['query'])
    get['class'] = clm63.labels_
    get_class = {}
    for i in range(0, max(clm63.labels_)+1):
        get_class[i] = get.loc[get['class'] == i]
    for i in range(len(get_class)):
        datatype = []
        index = get_class[i]['query'].str.len().idxmax()
        query = get_class[i]['query'][index]
        print(query)
        words = re.findall(r'\(.*?\)', query)
        print(words)
        print(len(words[0]))
        k = len(words[0])
        words = words[0][1:(k-1)]
        txt = words.split(',')
        for j in txt:
            if (j.isnumeric()):
                datatype.append('int')
            elif (validate_date(j)):
                datatype.append('date')
            else:
                datatype.append('str')
        fin_text = query.partition('VALUES')[0]
        fin_text = fin_text+' '+'VALUES'+' '+'{}'*len(txt)
        final_api.loc[i, 'text'] = fin_text
        final_api.loc[i, 'datatype'] = datatype
    return final_api


# Example to run the code:
data = pd.read_csv('/content/insert_used.csv')
data.dropna(inplace=True)
insert = data['QUERY']
insert = insert.to_list()
# So insert is a list with insert queries
final_api = generator_insert(insert)
final_api.to_csv('final_api.csv')

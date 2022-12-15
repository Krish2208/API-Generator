import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import StandardScaler
sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})


def read(path):
    df = pd.read_csv(path)
    row,column = df.shape
    df.drop(labels='DATE', axis= 1, inplace=True)
    df.drop(labels='TIME', axis= 1, inplace=True)
    df.drop(labels='QUERY', axis= 1, inplace=True)
    df.drop(labels='LOG', axis= 1, inplace=True)
    return df,row,column


def split(dataframe,row):
    j=0
    df_train = pd.DataFrame(columns = ["day","month","year","hour","min","sec"])
    for i in range(row):
        str1 = str(dataframe.iloc[i,0])
        lst = str1.split()
        df_train.loc[j] = lst
        j=j+1
    return df_train


def count_data(row,df_train):
    lst = []
    list_count= []
    count = 0
    ls1 = ["ab"]
    for i in range(row):
        time_str = str(df_train.iloc[i,0])+" "+str(df_train.iloc[i,1])+" "+str(df_train.iloc[i,2])+" "+str(df_train.iloc[i,3])+" "+str(df_train.iloc[i,4])
        if (i == 0):
            count = 0
            ls1[0] = (time_str)
        if(ls1[0] == (time_str)):
            count = count + 1
        if(ls1[0] != (time_str))or(i==(row-1)):
            ls1.append(count)
            lst.append(ls1)
            ls_count = [(df_train.iloc[(i-1),0]),(df_train.iloc[(i-1),1]),(df_train.iloc[(i-1),2]),(df_train.iloc[(i-1),3]),df_train.iloc[(i-1),4],count]
            list_count.append(ls_count)
            count = 1
            ls1 = ["ab"]
            ls1[0] = str(df_train.iloc[i,0])+" "+str(df_train.iloc[i,1])+" "+str(df_train.iloc[i,2])+" "+str(df_train.iloc[i,3])+" "+(str(df_train.iloc[i,4]))

    df_count = pd.DataFrame(data = list_count, columns =["day","month","year","hour","min","count"])
    
    return df_count


#Here the prediction function comes

def partition_dataset(sequence_length, data,index_count):
    x,y=[],[]
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_count])

    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x,y

def predict(df_count,model):
    index_count = df_count.columns.get_loc("count")
    sequence_length = 2
    nrows = df_count.shape[0]
    np_data_unscaled = np.array(df_count)
    np_data_unscaled = np.reshape(np_data_unscaled, (nrows, -1))
    x,y = partition_dataset(sequence_length,np_data_unscaled,index_count)
    scaler1= StandardScaler()
    scaler4= StandardScaler()
    x_test = scaler1.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    y_pred = scaler4.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)

    #prediction section
    y_pred = model.predict(x_test)
    y_pred= y_pred.reshape((y_pred.shape[0],))
    
    y_pred_unscaled = scaler4.inverse_transform(y_pred.reshape(1,-1))
    return y_pred_unscaled

def main(path):
    model = load_model("Time_series_model.h5")
    df,row,column = read(path)
    df_train = split(df,row)
    row,column = df_train.shape
    df_count = count_data(row,df_train)
    y_pred = predict(df_count,model)
    y_pred_min = df_count['min']
    print(y_pred)
    return (list(df_count["min"]),list(df_count["count"]))


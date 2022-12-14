import math
from pathlib import Path
import matplotlib.pyplot as plt
from warnings import simplefilter
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from math import sqrt

from datetime import date, timedelta # Date Functions
import matplotlib.dates as mdates # Formatting dates
from sklearn.metrics import mean_absolute_error, mean_squared_error # For measuring model performance / errors
from sklearn.preprocessing import MinMaxScaler #to normalize the price data 
from keras.models import Sequential # Deep learning library, used for neural networks
from keras.layers import LSTM, Dense # Deep learning classes for recurrent and regular densely-connected layers
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})


#functions are below
def read(path):
    df = pd.read_csv(path)
    row,column = df.shape
    df.drop(labels='Unnamed: 1', axis= 1, inplace=True)
    return df,row,column

def split(dataframe,row):
    j=0
    dct = {"Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3}
    df_train = pd.DataFrame(columns = ["day","month","year","hour","min","sec"])
    for i in range(row):
        str1 = str(dataframe.iloc[i,0])
        lst = str1.split()
        if len(lst) == 6:
            for k in range(len(lst)):
                if lst[k] in dct:
                    lst[k] = dct[lst[k]]
                lst[k] = int(lst[k])
            df_train.loc[j] = lst
            j=j+1
        else:
            continue
  
    return df_train

def count_data(row,df_train):
    lst = []
    list_count= []
    count = 0
    ls1 = ["ab"]
    for i in range(row):
        time_str = str(df_train.iloc[i,0])+" "+str(df_train.iloc[i,1])+" "+str(df_train.iloc[i,2])+" "+str(df_train.iloc[i,3])
        if (i == 0):
            count = 0
            ls1[0] = (time_str)
        if(ls1[0] == (time_str)):
            count = count + 1
        if(ls1[0] != (time_str))or(i==(row-1)):
            ls1.append(count)
            lst.append(ls1)
            ls_count = [(df_train.iloc[(i-1),0]),(df_train.iloc[(i-1),1]),(df_train.iloc[(i-1),2]),(df_train.iloc[(i-1),3]),count]
            list_count.append(ls_count)
            count = 1
            ls1 = ["ab"]
            ls1[0] = (str(df_train.iloc[i,0])+" "+str(df_train.iloc[i,1])+" "+str(df_train.iloc[i,2])+" "+str(df_train.iloc[i,3]))
  
    df_count = pd.DataFrame(data = list_count, columns =["day","month","year","hour","count"])
    return df_count

def cleaning(df_count):
    mean_value = df_count['count'].mean()
    df_count=df_count[df_count['count']<(9*mean_value)]
    df_count.dropna(inplace = True)
    return df_count

def preprocessing(df_count,sequence_length):
    # Get the number of rows in the data
    nrows = df_count.shape[0]

    # Convert the data to numpy values
    np_data_unscaled = np.array(df_count)
    np_data_unscaled = np.reshape(np_data_unscaled, (nrows, -1))

    # Prediction Index
    index_count = df_count.columns.get_loc("count")
    
    # Split the training data into train and train data sets
    # As a first step, we get the number of rows to train the model on 80% of the data 
    train_data_len = math.ceil(np_data_unscaled.shape[0] * 0.8)

    # Create the training and test data
    train_data = np_data_unscaled[0:train_data_len, :]
    test_data = np_data_unscaled[train_data_len - sequence_length:, :]

    return train_data,test_data,index_count,train_data_len

def partition_dataset(sequence_length,data,index_count):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_count]) #contains the prediction values for validation (3rd column = Close),  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

def create_data(sequence_length,train_data,test_data):
  # Generate training data and test data
  x_train_un, y_train_un = partition_dataset(sequence_length, train_data)
  x_test_un, y_test_un = partition_dataset(sequence_length, test_data)
  scaler1= StandardScaler()
  scaler2= StandardScaler()
  scaler3= StandardScaler()
  scaler4= StandardScaler()
  x_train = scaler1.fit_transform(x_train_un.reshape(-1, x_train_un.shape[-1])).reshape(x_train_un.shape)
  x_test = scaler3.fit_transform(x_test_un.reshape(-1, x_test_un.shape[-1])).reshape(x_test_un.shape)
  y_train = scaler2.fit_transform(y_train_un.reshape(-1, y_train_un.shape[-1])).reshape(y_train_un.shape)
  y_test = scaler4.fit_transform(y_test_un.reshape(-1, y_test_un.shape[-1])).reshape(y_test_un.shape)

  return x_train,x_test,y_train,y_test,scaler4

def model_creation(epochs,batch_size,x_train):
# Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
    n_neurons = x_train.shape[1] * x_train.shape[2]
    model = Sequential()
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], 5)))
    model.add(LSTM(n_neurons, return_sequences=False))
    model.add(Dense(5))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model,epochs,batch_size

def prediction(model,x_test,y_test,scaler4):
    # Get the predicted values
    y_pred = model.predict(x_test)
    y_pred= y_pred.reshape((130,))
    y_pred_unscaled = scaler4.inverse_transform(y_pred.reshape(1,-1))
    y_test_unscaled = scaler4.inverse_transform(y_test.reshape(1,-1))
    return y_pred_unscaled,y_test_unscaled

def error_analysis(y_test_unscaled, y_pred_unscaled):
  # Mean Absolute Error (MAE)
  MAE = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
  MAE_value = np.round(MAE, 2)

  # Mean Absolute Percentage Error (MAPE)
  MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred_unscaled)/ y_test_unscaled))) * 100
  MAPE_value = np.round(MAPE, 2)

  # Median Absolute Percentage Error (MDAPE)
  MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred_unscaled)/ y_test_unscaled)) ) * 100
  MDAPE_value = np.round(MDAPE, 2)

  return MAE_value,MAPE_value,MDAPE_value

def plotting(df_valid_pred):
    # Create the lineplot
    fig, ax1 = plt.subplots(figsize=(32, 5), sharex=True)
    ax1.tick_params(axis="x", rotation=0, labelsize=10, length=0)
    plt.title("Predictions vs Ground Truth")
    sns.lineplot(data=df_valid_pred)
    plt.show()

## Main function runs below
def main(path,path_to_save):
    #path = "/content/gdrive/MyDrive/time_series_forecasting/weblog_final_edit.csv"
    df,row,column = read(path)
    df_train = split(df)
    row, column = df_train.shape
    df_count = count_data(row,df_train)
    df_count = cleaning(df_count)
    sequence_length = 110
    train_data,test_data,index_count,train_data_len = preprocessing(df_count,sequence_length = sequence_length)
    x_train,x_test,y_train,y_test,scaler4 = create_data(sequence_length = sequence_length,train_data = train_data,test_data = test_data)
    model,epochs,batch_size = model_creation(20,1,x_train)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    y_pred_unscaled,y_test_unscaled = prediction(model,x_test,y_test,scaler4)
    MAE,MAPE,MDAPE = error_analysis(y_test_unscaled, y_pred_unscaled)
    y_pred_unscaled= y_pred_unscaled.reshape(130)
    df_valid_pred = df_count[train_data_len:]
    df_valid_pred['y_pred']= y_pred_unscaled
    plotting(df_valid_pred)
    model.save_weights(path_to_save)
    return 0
##try to put a try and except command

main()
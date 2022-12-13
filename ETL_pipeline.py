# Extraction, Transformation and Loading pipeline
from analysis import get_csv
import pandas as pd
from cluster_SQL import splitter,gen_module
from mysql_connector import mysql_connection

class ETL:
    def __init__(self):
        self.variables = 0;

    def analysis_module(self,file):

        """
        analysis module is being used
        for getting the csv file after
        analysing the whole log file.

        function called: 
        ------------
            get_csv from analysis.py

        paramenters: 
        ------------
            file object: log file

        returns: 
        ------------
            csv file: derived from the log file using ner model
        """
        file.save('./static/mysql.log')
        with open('./static/mysql.log','r') as log_file:
            print(log_file)
            csv_file = get_csv(log_file)
            # df = pd.read_csv(csv_file)
        csv_file.to_csv('./static/mysql.csv')
        return csv_file

    def countQueries(self):
        df = pd.read_csv(r'D:\projects\API\api-backend\static\mysql.csv', squeeze=False,header=0)
        print(type(df),"2")
        splits = splitter(df)
        return splits

    def getQueriesSQL(self):
        etl = ETL()
        splits = etl.countQueries()
        for i in splits:
            try:
                final_api= gen_module(i)
                print(final_api)
            except ValueError:
                print('No values')
        final_api= pd.DataFrame(final_api)
        print(final_api)
        final_api.to_csv('./static/final.csv',index=False,header=True)
        return final_api

    def createNameQuery(self):
        df = pd.read_csv('./static/final.csv')
        queries = df['text']
        names = []
        for i in queries:
            i = i.lower()
            etl = ETL()
            i = etl.removePunct(i)
            if '*' in i:
                i = i.replace('*','All')
            list_of_words = i.split()
            if 'from' in list_of_words:
                list_of_words[list_of_words.index('from')]='From'
            if list_of_words[0] == 'select':
                etl = ETL()
                name,detail = etl.select(list_of_words)
                names = []
                name_dict = {}
                name_dict['name'] = name
                name_dict['details'] = detail
                names.append(name_dict)
                print(names)
        return names 
                
    def select(self,words):
        select_dict={}
        first = []
        second = []
        third = []
        for i in words[1::]:
            if '.' in i:
                index = words.index(i)
                i=i[i.index('.')+1::]
                words[index] = i
                print(i)
            if i=='From':
                index = words.index(i)
                break;
            else:
                index = words.index(i)
                first.append(i)
        
        for i in words[index+1::]:
            if i == 'where':
                index = words.index(i)
                break;
            else:
                index = words.index(i)
                second.append(i)
        
        for i in words[index+1::]:
            third.append(i)
        
        select_dict['select'] = first
        select_dict['from'] = second
        select_dict['where'] = third

        name = 'read'
        
        if 'as' in first:
            del first[first.index('as')-1]
            del first[first.index('as')]
        
        if len(first)>1:
            name = name + 'Details' 
        else:
            name = name + first[0].capitalize()
        if len(third)!=0:
            try:
                name = name +'By'+ third[third.index('{}')-2].capitalize()
            except ValueError:
                name = name +'By'+ third[third.index('=')-1].capitalize()

        return name,first

    
    def removePunct(self,string):
        punc = "',!()-[];:\,/?@#$%^&*_~"
        for ele in string:
            if ele in punc:
                string = string.replace(ele, "")
        return string
        
    # def executeAPISQL(self,id,username,password,host_url,database,port):
    #     df = pd.read_csv("./static/file.csv")
    #     query = df[id]
    #     mysql_connection(username,password,database,host_url,query,port)
        
    # def executeAPINoSQL(self,id,username,password,host_url,database,port):
    #     a=0

etl = ETL()
etl.createNameQuery()

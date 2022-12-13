from pymongo import MongoClient

def mongo_connection(host,port,database,query):
    client = MongoClient(host=host, port=port)
    db = client('{database}')
    

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from requests import post
from flask_api import status
import pandas as pd
# from mysql import mysql_connection
from ETL_pipeline import ETL

app = Flask('__name__')
CORS(app)

@app.route('/file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        etl = ETL()
        etl.analysis_module(file)
        return 'success', status.HTTP_200_OK

@app.route('/numerics', methods=['GET'])
def analysis():
    # import the module 
    etl = ETL()
    splits = etl.countQueries()
    create,read,update,delete,miscell = splits[3],splits[1],splits[0],splits[2],splits[4]
    numeric = [len(create),len(read),len(update),len(delete),len(miscell)]
    return {'numeric': numeric}

# @app.route('/table', methods=['GET'])
# def tables():
#     # import the module
#     tables = 'table1'
#     return {'tables':tables}

@app.route('/time', methods=['GET'])
def time():
    # import the module
    create = ['create time series']
    read = ['read time series']
    update = ['update time series']
    delete = ['delete time series']
    return {'create':create, 'read':read,'update':update,'delete':delete}


@app.route('/api', methods=['GET'])
def api():
    # import the module
    etl = ETL()
    query = etl.getQueriesSQL()
    return {'api':"success"}

@app.route('/database', methods=['POST'])
def database():
    if request.method == 'POST':
        info = json.loads(request.data)
        username = info.get('username')
        password = info.get('password')
        database = info.get('database')
        platform = info.get('platform')
        host_url = info.get('host_url')
        query_input = info.get('query_input')
        
        if platform == 'mysql':
            etl=ETL()
            etl.executeAPISQL(query_input,username,password,host_url,database)
        if platform == 'nosql':
            etl=ETL()
            etl.executeAPINoSQL()

#         return {'username': username, 'password':password, 'database':database, 'platform':platform}, status.HTTP_201_CREATED
#     else:
#         return status.HTTP_400_BAD_REQUEST



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
    
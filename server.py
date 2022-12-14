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


@app.route('/time', methods=['GET'])
def time():
    etl=ETL()
    count = etl.predict()
    return {'count':count}


@app.route('/api', methods=['GET'])
def api():
    # import the module
    etl = ETL()
    query = etl.getQueriesSQL()
    return {'api':"success"}

@app.route('/connection', methods=['POST'])
def connection():
    if request.method == 'POST':
        info = json.loads(request.data)
        username = info.get('username')
        password = info.get('password')
        database = info.get('database')
        host_url = info.get('host_url')
        query_name = info.get('query_name')
        query_info = info.get('query_info')
        port=info.get('port')
        etl=ETL()
        # etl.dfconcat()
        output = etl.executeAPISQL(username,password,host_url,database,query_name,port,query_info)
        return {"result": output}, status.HTTP_200_OK
    else:
        return status.HTTP_400_BAD_REQUEST

@app.route('/getapis', methods=['GET'])
def api_details():
    # import the module
    etl = ETL()
    apis = etl.getAPIs()
    return {'apis': apis}


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
    
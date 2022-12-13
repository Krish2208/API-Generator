import mysql.connector

def mysql_connection(username,password,database,host_url,queries,port):
    connect = mysql.connector.connect(
        user=username,
        password=password,
        database=database,
        host=host_url,
        port=port
    )
    cursor = connect.cursor()
    for i in queries:
        cursor.execute(str(i))
    connect.commit()
    connect.close()


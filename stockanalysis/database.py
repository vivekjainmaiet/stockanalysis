from pickle import NONE
import mysql.connector as connection
from mysql.connector.constants import ClientFlag
import pandas as pd
from param import config


class MySQLDB:


    def getConnection(self):
        conn=connection.connect(**config)
        self.connection = conn
        return conn;

    def closeConnection(self):
        self.connection.close()

    def getData_df(self,query):
        try:
            conn = self.getConnection()
            #query = "Select * from studentdetails;"
            result_dataFrame = pd.read_sql(query,conn)
            conn.close() #close the connection
        except Exception as e:
            self.connection.close()
            print(str(e))
        return result_dataFrame
    #insert or updte for single record or single dictionary
    #https://dev.mysql.com/doc/connector-python/en/connector-python-example-cursor-transaction.html
    def InsertUpdateData(self,query,data):
        conn=self.getConnection()
        _cursor = conn.cursor()
        # data could be list or dictionary
        _cursor.execute(query, data)
        _cursor.close()
        conn.commit()  # and commit changes
        conn.close()
    #Create a object like table or view etc
    def ExecuteDDLStatement(self,createStatement):
        _conn = self.getConnection()
        _cursor = _conn.cursor()  # initialize connection cursor
        #createStatement = "CREATE TABLE stocksdb.tet ( ID INT NOT NULL, Date date NOT NULL)"
        _cursor.execute(createStatement)  # create a new 'testdb' database
        _cursor.close()
        _conn.commit()
        _conn.close()  # close connection

    #Save dataframe to table
    #https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-executemany.html
    def SaveDFToTable(self,query,df):
        conn = self.getConnection()
        _cursor = conn.cursor()
        _cursor.executemany(query, df.to_numpy().tolist())
        _cursor.close()
        conn.commit()  # and commit changes
        conn.close()

    def SaveDictionayToTable(self,query,dataDict):
        conn = self.getConnection()
        _cursor = conn.cursor()
        df = pd.DataFrame.from_dict(dataDict)
        listdata=df.to_numpy().tolist()
        print(listdata)
        #df.to_sql(TableName,conn)
        _cursor.executemany(query, listdata)
        _cursor.close()
        conn.commit()  # and commit changes
        conn.close()

if __name__ == "__main__":
    mysql=MySQLDB()
    df=mysql.getData_df("SELECT * FROM stocksdb.StocksList;")
    print(df)
    query = ("INSERT INTO stocksdb.StocksList ( StockName,StockCode,exchange) "
             "VALUES (%s, %s, %s)")
    #insert single value from list of dictionary
    #data=('Tata Motors Ltd','TATAMOTORS','NSE')
    #data = {'StockName': 'Tata Motors Ltd','StockCode':'TATAMOTORS','exchange':'NSE'}
    #mysql.InsertUpdateData(query, data)
    queryDic = ("INSERT INTO stocksdb.StocksList ( StockName,StockCode,exchange) "
             "VALUES (%(StockName)s, %(StockCode)s, %(exchange)s)")
    #data=('Tata Motors Ltd','TATAMOTORS','NSE')


    #save the dataframe to table
    data = pd.DataFrame([('Tata Motors Ltd', 'TATAMOTORS', 'NSE'),
           ('Microsoft', 'Micro', 'NASDAQ')])
    mysql.SaveDFToTable(query, data)


    #save data from multple value from dictionary to table
    dataDictstocks = {
        'StockName': ['Tata Motors Ltd1','Microsoft1'],
        'StockCode': ['TATAMOTORS1','Micro1'],
        'exchange': ['NSE','NASDAQ']
    }
    #mysql.SaveDictionayToTable(query, dataDictstocks)


    #mysql.ExecuteDDLStatement("CREATE TABLE stocksdb.tet ( ID INT NOT NULL, Date date NOT NULL)")

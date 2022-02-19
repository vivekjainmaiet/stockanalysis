from pickle import NONE
import mysql.connector as connection
import pandas as pd
from param import config
from stockanalysis.database import *

mysql=MySQLDB()
df=mysql.getData_df("SELECT * FROM stocksdb.StocksList;")
SaveDFToTable(self, query, df)

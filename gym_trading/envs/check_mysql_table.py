import mysql.connector
from Config import Config

conf = Config()

mydb = mysql.connector.connect(
  host= conf.MYSQL_HOST,
  user=conf.MYSQL_USER,
  passwd=conf.MYSQL_PASSWORD,
  database=conf.MYSQL_DATABASE,
  port=conf.MYSQL_PORT
)

mycursor = mydb.cursor()

table = conf.TICKER + '_extended_100000' # Write here the name of the MySQL table you want to check/list

mycursor.execute('SELECT * FROM '+ table)

myresult = mycursor.fetchall()

for x in myresult:
  print(x)
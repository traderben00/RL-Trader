from Config import Config
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import DECIMAL, DATE, TIME, INT

conf = Config()

engine = create_engine('mysql+mysqldb://'+conf.MYSQL_USER+':'+conf.MYSQL_PASSWORD+'@'+conf.MYSQL_HOST_PORT+'/'+conf.MYSQL_DATABASE)

instrument = conf.TICKER + '_extended_100000' # # Write here the name of the MySQL table you want to fill

df_raw = pd.read_csv('datasets/raw_dataset/'+instrument+'.csv', sep=';')

df = df_raw.dropna()

df.to_sql(con=engine, name=instrument, if_exists='replace', index_label='ID', chunksize=10000, dtype={'DATE' : DATE(),'TIME': TIME(),'OPEN': DECIMAL(6,5), 'HIGH': DECIMAL(6,5), 'LOW' : DECIMAL(6,5), 'CLOSE' : DECIMAL(6,5), 'TICKVOL' : INT(), 'RETURN1' : DECIMAL(6,5), 'RETURN5' : DECIMAL(6,5), 'RETURN15' : DECIMAL(6,5), 'RETURN60' : DECIMAL(6,5), 'RETURN240' : DECIMAL(6,5), 'RETURN1440' : DECIMAL(6,5), 'RETURN4320' : DECIMAL(6,5), 'PCTL15' : DECIMAL(6,5), 'PCTL60' : DECIMAL(6,5), 'PCTL240' : DECIMAL(6,5), 'PCTL1440' : DECIMAL(6,5), 'PCTL4320' : DECIMAL(6,5), 'MA200DIF' : DECIMAL(6,5), 'MA1260DIF' : DECIMAL(6,5), 'MA3000DIF' : DECIMAL(6,5), 'ATR14' : DECIMAL(6,5), 'ATR70' : DECIMAL(6,5), 'ATR210' : DECIMAL(6,5), 'ATR840' : DECIMAL(6,5), 'ATR3360' : DECIMAL(6,5), 'RSI14' : DECIMAL(4,2), 'RSI70' : DECIMAL(4,2), 'RSI210' : DECIMAL(4,2), 'RSI840' : DECIMAL(4,2), 'RSI3360' : DECIMAL(4,2)})

engine.execute('SELECT * FROM '+instrument).fetchall()

print('Filling was successful')
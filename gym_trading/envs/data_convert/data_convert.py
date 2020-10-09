import pandas as pd
import time
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import gc

from Config import Config

conf = Config()

#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)

instrument = conf.TICKER
data_path = instrument+'_example_100000.csv'
rounding_decimal = 5

data_raw = pd.read_csv(data_path, sep=';')
data = data_raw['CLOSE']
data_raw = data_raw.drop(labels={'VOL', 'SPREAD' }, axis=1)
#======================================================

# Calculating ATR

ATR_period = 14

def true_range(current_high, current_low, previous_close=0): # previous close = 0 if no prior data
    return max(abs(current_high - current_low),
               abs(current_high - previous_close),
               abs(current_low  - previous_close))

class ATR:

    def __init__(self):
        self.true_range_history_14 = []
        self.true_range_history_70 = []
        self.true_range_history_210 = []
        self.true_range_history_840 = []
        self.true_range_history_3360 = []
        self.prior_ATR_14 = -1 # -1 not possible, represents nonexistence of ATR
        self.prior_ATR_70 = -1 # -1 not possible, represents nonexistence of ATR
        self.prior_ATR_210 = -1 # -1 not possible, represents nonexistence of ATR
        self.prior_ATR_840 = -1 # -1 not possible, represents nonexistence of ATR
        self.prior_ATR_3360 = -1 # -1 not possible, represents nonexistence of ATR

    def current_ATR_14(self, current_true_range, period):
        if len(self.true_range_history_14) < period: # until required number of periods has been met
            self.true_range_history_14.append(current_true_range) # store true ranges
            current_ATR = -1 # return -1 to show that no ATR could be calculated
        
        elif self.prior_ATR_14 == -1: # required number of periods met, no prior ATR
            current_ATR = sum(self.true_range_history_14) / period
            
        else: # required number of periods met, prior ATR
            current_ATR = ((self.prior_ATR_14 * (period - 1)) + current_true_range) / period

        self.prior_ATR_14 = current_ATR
        return current_ATR
    
    def current_ATR_70(self, current_true_range, period):
        if len(self.true_range_history_70) < period: # until required number of periods has been met
            self.true_range_history_70.append(current_true_range) # store true ranges
            current_ATR = -1 # return -1 to show that no ATR could be calculated
        
        elif self.prior_ATR_70 == -1: # required number of periods met, no prior ATR
            current_ATR = sum(self.true_range_history_70) / period
            
        else: # required number of periods met, prior ATR
            current_ATR = ((self.prior_ATR_70 * (period - 1)) + current_true_range) / period

        self.prior_ATR_70 = current_ATR
        return current_ATR
    
    def current_ATR_210(self, current_true_range, period):
        if len(self.true_range_history_210) < period: # until required number of periods has been met
            self.true_range_history_210.append(current_true_range) # store true ranges
            current_ATR = -1 # return -1 to show that no ATR could be calculated
        
        elif self.prior_ATR_210 == -1: # required number of periods met, no prior ATR
            current_ATR = sum(self.true_range_history_210) / period
            
        else: # required number of periods met, prior ATR
            current_ATR = ((self.prior_ATR_210 * (period - 1)) + current_true_range) / period

        self.prior_ATR_210 = current_ATR
        return current_ATR
    
    def current_ATR_840(self, current_true_range, period):
        if len(self.true_range_history_840) < period: # until required number of periods has been met
            self.true_range_history_840.append(current_true_range) # store true ranges
            current_ATR = -1 # return -1 to show that no ATR could be calculated
        
        elif self.prior_ATR_840 == -1: # required number of periods met, no prior ATR
            current_ATR = sum(self.true_range_history_840) / period
            
        else: # required number of periods met, prior ATR
            current_ATR = ((self.prior_ATR_840 * (period - 1)) + current_true_range) / period

        self.prior_ATR_840 = current_ATR
        return current_ATR
    
    def current_ATR_3360(self, current_true_range, period):
        if len(self.true_range_history_3360) < period: # until required number of periods has been met
            self.true_range_history_3360.append(current_true_range) # store true ranges
            current_ATR = -1 # return -1 to show that no ATR could be calculated
        
        elif self.prior_ATR_3360 == -1: # required number of periods met, no prior ATR
            current_ATR = sum(self.true_range_history_3360) / period
            
        else: # required number of periods met, prior ATR
            current_ATR = ((self.prior_ATR_3360 * (period - 1)) + current_true_range) / period

        self.prior_ATR_3360 = current_ATR
        return current_ATR

def fill_RSI_column(dataframe, n=14):

    delta = dataframe['CLOSE'].diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(n).mean()
    RolDown = dDown.rolling(n).mean().abs()

    RS = RolUp / RolDown
    rsi= 100.0 - (100.0 / (1.0 + RS))
    rsi = round(rsi, 2)
    dataframe['RSI'+ str(n)] = rsi



# this should be before the main loop:
ATR_object = ATR()

# Initialize the new columns
data_raw['RETURN1'] = None
data_raw['RETURN5'] = None
data_raw['RETURN15'] = None
data_raw['RETURN60'] = None
data_raw['RETURN240'] = None
data_raw['RETURN1440'] = None
data_raw['RETURN4320'] = None
data_raw['PCTL15'] = None
data_raw['PCTL60'] = None
data_raw['PCTL240'] = None
data_raw['PCTL1440'] = None
data_raw['PCTL4320'] = None
data_raw['MA200DIF'] = None
data_raw['MA1260DIF'] = None
data_raw['MA3000DIF'] = None
data_raw['ATR14'] = None
data_raw['ATR70'] = None
data_raw['ATR210'] = None
data_raw['ATR840'] = None
data_raw['ATR3360'] = None
data_raw['RSI14'] = None
data_raw['RSI70'] = None
data_raw['RSI210'] = None
data_raw['RSI840'] = None
data_raw['RSI3360'] = None

fill_RSI_column(data_raw,14)
fill_RSI_column(data_raw,14*5)
fill_RSI_column(data_raw,14*15)
fill_RSI_column(data_raw,14*60)
fill_RSI_column(data_raw,14*240)

for index in range(0,data.size):
    if index % 10000 == 0:
        if index == 0:
            start_time = time.time()
        else:
            end_time = time.time()
            print('For loop reached line number: ', str(index))
            print('10000 rows took: ' + str(end_time-start_time) + ' seconds')
            start_time = time.time()

    if index > 0:
        now_close_data = data_raw.iat[index, data_raw.columns.get_loc('CLOSE')]
        close_data_1 = data_raw.iat[index-1, data_raw.columns.get_loc('CLOSE')]
        return_final = (now_close_data - close_data_1) / close_data_1
        data_raw.at[index, 'RETURN1'] = round(return_final, 4)
    if index > 4:
        close_data_5 = data_raw.iat[index-5, data_raw.columns.get_loc('CLOSE')]
        return_final = (now_close_data - close_data_5) / close_data_5
        data_raw.at[index, 'RETURN5'] = round(return_final, 4)
    if index > 14:
        close_data_15 = data_raw.iat[index-15, data_raw.columns.get_loc('CLOSE')]
        return_final = (now_close_data - close_data_15) / close_data_15 
        data_raw.at[index, 'RETURN15'] = round(return_final, 4)
        closeDataframe = pd.DataFrame(data=data.values[index-15:index])
        closePctl = closeDataframe.rank(pct=True).iloc[-1]
        data_raw.at[index, 'PCTL15'] = round(closePctl.ix[0, 1], 4)
    if index > 59:
        close_data_60 = data_raw.iat[index-60, data_raw.columns.get_loc('CLOSE')]
        return_final = (now_close_data - close_data_60) / close_data_60
        data_raw.at[index, 'RETURN60'] = round(return_final, 4)
        closeDataframe = pd.DataFrame(data=data.values[index-60:index])
        closePctl = closeDataframe.rank(pct=True).iloc[-1]
        data_raw.at[index, 'PCTL60'] = round(closePctl.ix[0, 1], 4)
    if index > 199:
        rolling_data = data[index-200:index]
        rolling_mean = rolling_data.rolling(window=200).mean()
        rolling_mean = round(rolling_mean.values[-1], rounding_decimal)
        data_raw.at[index, 'MA200DIF'] = round(data[index] - rolling_mean, rounding_decimal)
    if index > 239:
        close_data_240 = data_raw.iat[index-240, data_raw.columns.get_loc('CLOSE')]
        return_final = (now_close_data - close_data_240) / close_data_240
        data_raw.at[index, 'RETURN240'] = round(return_final, 4)
        closeDataframe = pd.DataFrame(data=data.values[index-240:index])
        closePctl = closeDataframe.rank(pct=True).iloc[-1]
        data_raw.at[index, 'PCTL240'] = round(closePctl.ix[0, 1], 4)
    if index > 1259:
        rolling_data = data[index-1260:index]
        rolling_mean = rolling_data.rolling(window=1260).mean()
        rolling_mean = round(rolling_mean.values[-1], rounding_decimal)
        data_raw.at[index, 'MA1260DIF'] = round(data[index] - rolling_mean, rounding_decimal)
    if index > 1439:
        close_data_1440 = data_raw.iat[index-1440, data_raw.columns.get_loc('CLOSE')]
        return_final = (now_close_data - close_data_1440) / close_data_1440
        data_raw.at[index, 'RETURN1440'] = round(return_final, 4)
        closeDataframe = pd.DataFrame(data=data.values[index-1440:index])
        closePctl = closeDataframe.rank(pct=True).iloc[-1]
        data_raw.at[index, 'PCTL1440'] = round(closePctl.ix[0, 1], 4)
    if index > 2999:
        rolling_data = data[index-3000:index]
        rolling_mean = rolling_data.rolling(window=3000).mean()
        rolling_mean = round(rolling_mean.values[-1], rounding_decimal)
        data_raw.at[index, 'MA3000DIF'] = round(data[index] - rolling_mean, rounding_decimal)
    if index > 4319:
        close_data_4320 = data_raw.iat[index-4320, data_raw.columns.get_loc('CLOSE')]
        return_final = (now_close_data - close_data_4320) / close_data_4320
        data_raw.at[index, 'RETURN4320'] = round(return_final, 4)
        closeDataframe = pd.DataFrame(data=data.values[index-4320:index])
        closePctl = closeDataframe.rank(pct=True).iloc[-1]
        data_raw.at[index, 'PCTL4320'] = round(closePctl.ix[0, 1], 4)

    prev_close = 0 if index == 0 else data_raw['CLOSE'][index-1]
    current_true_range = true_range(data_raw['HIGH'][index],data_raw['LOW'][index],prev_close) # arguments need to be added
    current_ATR_14 = ATR_object.current_ATR_14(current_true_range, ATR_period)
    current_ATR_70 = ATR_object.current_ATR_70(current_true_range, ATR_period*5)
    current_ATR_210 = ATR_object.current_ATR_210(current_true_range, ATR_period*15)
    current_ATR_840 = ATR_object.current_ATR_840(current_true_range, ATR_period*60)
    current_ATR_3360 = ATR_object.current_ATR_3360(current_true_range, ATR_period*240)

    if current_ATR_14 != -1: # if there is sufficient history for a current ATR to be calculated:
        data_raw.at[index, 'ATR14'] = round(current_ATR_14, rounding_decimal)
    if current_ATR_70 != -1: # if there is sufficient history for a current ATR to be calculated:
        data_raw.at[index, 'ATR70'] = round(current_ATR_70, rounding_decimal)
    if current_ATR_210 != -1: # if there is sufficient history for a current ATR to be calculated:
        data_raw.at[index, 'ATR210'] = round(current_ATR_210, rounding_decimal)
    if current_ATR_840 != -1: # if there is sufficient history for a current ATR to be calculated:
        data_raw.at[index, 'ATR840'] = round(current_ATR_840, rounding_decimal)
    if current_ATR_3360 != -1: # if there is sufficient history for a current ATR to be calculated:
        data_raw.at[index, 'ATR3360'] = round(current_ATR_3360, rounding_decimal)
    

    if index == 0:
        data_raw.iloc[index:index].to_csv(instrument+'_converted_100000.csv', sep=';', header={'DATE','TIME','OPEN','HIGH','LOW','CLOSE','TICKVOL', 'RETURN1', 'RETURN5', 'RETURN15', 'RETURN60', 'RETURN240', 'RETURN1440', 'RETURN4320', 'PCTL15', 'PCTL60', 'PCTL240', 'PCTL1440', 'PCTL4320', 'MA200DIF', 'MA1260DIF', 'MA3000DIF', 'ATR14', 'ATR70', 'ATR210', 'ATR840', 'ATR3360', 'RSI14', 'RSI70', 'RSI210', 'RSI840', 'RSI3360'}, index=False, mode='a')
    if index > 4320:
        data_raw.iloc[index-1:index].to_csv(instrument+'_converted_100000.csv', sep=';', header=False, index=False, mode='a')

    gc.collect()


print('Converted Data:\n')
print(data_raw)

# Plot
print('\nPlot data...')

plt.plot(data_raw['CLOSE'], color='black', label='true', linewidth=1)

plt.title('Dataset')
plt.ylabel('Close price')
plt.xlabel('Time')

plt.show()
plt.gcf().clear()

print('DONE')
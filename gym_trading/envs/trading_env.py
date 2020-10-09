import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import time
import numpy as np
from numpy import random
import pandas as pd
import logging
import pdb
import tempfile
from sqlalchemy import create_engine

from sklearn.preprocessing import MinMaxScaler


from Config import Config

conf = Config()

log = logging.getLogger(__name__)
log.info('%s logger started.',__name__)


def load_data(instrument, train):
  if train:
    data_path = conf.TRAINING_DATA_PATH
    csv_path = os.path.join(data_path, instrument + conf.csv_file)
    if conf.num_of_rows_read > 0:
      return pd.read_csv(csv_path, sep=';', nrows=conf.num_of_rows_read)
    else:
      return pd.read_csv(csv_path, sep=';')
  else:
    data_path = conf.INPUT_PREDICT_DATA_PATH
    csv_path = os.path.join(data_path, conf.TICKER + conf.input_predict_extension)
    if conf.num_of_rows_read > 0:
      return pd.read_csv(csv_path, sep=';')


current_instrument = conf.TICKER + '_extended_100000'

if conf.IS_MYSQL and conf.IS_TRAIN:
  engine = create_engine('mysql+mysqldb://'+conf.MYSQL_USER+':'+conf.MYSQL_PASSWORD+'@'+conf.MYSQL_HOST+'/'+conf.MYSQL_DATABASE)
  if conf.num_of_rows_read > 0:
    df_raw = pd.read_sql('SELECT * FROM '+current_instrument+' WHERE ID <= '+str(conf.num_of_rows_read), engine, index_col='ID')
  else:
    df_raw = pd.read_sql('SELECT * FROM '+current_instrument, engine, index_col='ID')
elif conf.IS_MYSQL == False and conf.IS_TRAIN:
  df_raw = load_data(current_instrument, train=True)    
elif conf.IS_TRAIN == False:
  df_raw = load_data(current_instrument, train=False)

df = df_raw.dropna()

DEFAULT_MINUTE_WINDOW = len(df.index) - 1


def _prices2returns(prices):
  px = pd.DataFrame(prices)
  nl = px.shift().fillna(0)
  R = ((px - nl)/nl).fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
  R = np.append( R[0].values, 0)
  return R
  
class EnvSrc(object):

  Name = 'REINFORCE_TRADER'

  def __init__(self, minutes=DEFAULT_MINUTE_WINDOW, name=Name, scale=False ,current_instrument=current_instrument,train=True):
    global data_length_per_cpu
    global data_length
    global df
    self.name = current_instrument
    self.minutes = minutes+1
        
    start_time = time.time()

    df = df.drop(axis=1, columns={'DATE', 'TIME', 'OPEN', 'HIGH', 'LOW'})
    df_raw_return = df['RETURN1']
    scaler = MinMaxScaler(copy=False)

    print('Data to be scaled:\n')
    print(df)

    df = scaler.fit_transform(df)
    
    end_time = time.time()
    

    print('Loading data and scaling took: (in seconds)')
    print(str(end_time - start_time))

    self.min_values = df.min(axis=0)
    self.max_values = df.max(axis=0)
    self.data = df
    self.df_raw_return = df_raw_return
    self.step = 0
    
  def reset(self):
    # we want contiguous data
    data_len = len(self.data) - 2
    self.idx = 0 #np.random.randint( low = 0, high=data_len)
    self.step = 0

  def _step(self):
    obs = self.data[self.idx]
    ret = self.df_raw_return[self.idx]
    self.idx += 1
    self.step += 1
    done = self.step >= self.minutes
    return obs,done, ret

class TradingSim(object) :
  ''' Implements core trading simulator for single-instrument univ '''

  def __init__(self, steps, trading_cost_bps = conf.execution_penalty, time_cost_bps = conf.timestep_penalty):
    # invariant for object life
    self.trading_cost_bps = trading_cost_bps
    self.time_cost_bps    = time_cost_bps
    self.steps            = steps
    # change every step
    self.step             = 0
    self.actions          = np.zeros(self.steps)
    self.navs             = np.ones(self.steps)
    self.mkt_nav          = np.ones(self.steps)
    self.strat_retrns     = np.ones(self.steps)
    self.posns            = np.zeros(self.steps)
    self.costs            = np.zeros(self.steps)
    self.trades           = np.zeros(self.steps)
    self.mkt_retrns       = np.zeros(self.steps)
    
  def reset(self):
    self.step = 0
    self.actions.fill(0)
    self.navs.fill(1)
    self.mkt_nav.fill(1)
    self.strat_retrns.fill(0)
    self.posns.fill(0)
    self.costs.fill(0)
    self.trades.fill(0)
    self.mkt_retrns.fill(0)
    
  def _step(self, action, retrn ):
    ''' Given an action and return for prior period, calculates costs, navs,
        etc and returns the reward and a summary of the step's activity. '''

    # BOD_POSN: If we don't have a trade open or step=0, it's gonna be 0, which means reward is 0 and the reward on first trade open = the trading cost only
    # If we have a short or long trade opened, the reward will be calculated according to the price difference
    #in the negative or positive way respectively (short/long) and no matter there are FLAT actions during the opened trade. 
    
    bod_posn = 0.0 if self.step == 0 else self.posns[self.step-1]
    bod_nav  = 1.0 if self.step == 0 else self.navs[self.step-1]
    mkt_nav  = 1.0 if self.step == 0 else self.mkt_nav[self.step-1]

    self.mkt_retrns[self.step] = retrn
    self.actions[self.step] = action
    
    self.posns[self.step] = bod_posn + action - 1
    self.trades[self.step] = action - 1
    
    trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps 
    self.costs[self.step] = trade_costs_pct +  self.time_cost_bps
    reward = ( (bod_posn * retrn) - self.costs[self.step] )

    self.strat_retrns[self.step] = reward

    if self.step != 0 :
      self.navs[self.step] =  bod_nav * (1 + self.strat_retrns[self.step-1])
      self.mkt_nav[self.step] =  mkt_nav * (1 + self.mkt_retrns[self.step-1])
    
    info = { 'reward': reward, 'nav':self.navs[self.step], 'costs':self.costs[self.step] }
    #print(info)
    self.step += 1      
    return reward, info

  def to_df(self):
    '''returns internal state in new dataframe '''
    cols = ['action', 'bod_nav', 'mkt_nav','mkt_return','sim_return',
            'position','costs', 'trade' ]
    rets = _prices2returns(self.navs)
    #pdb.set_trace()
    df = pd.DataFrame( {'action':     self.actions, # now's action (from agent)
                          'bod_nav':    self.navs,    # BOD Net Asset Value (NAV)
                          'mkt_nav':  self.mkt_nav, 
                          'mkt_return': self.mkt_retrns,
                          'sim_return': self.strat_retrns,
                          'position':   self.posns,   # EOD position
                          'costs':  self.costs,   # eod costs
                          'trade':  self.trades },# eod trade
                         columns=cols)
    return df

class TradingEnv(gym.Env):
  '''This gym implements a simple trading environment for reinforcement learning.

  SELL (0)
  FLAT (1)
  BUY (2)

  At the beginning of your episode, you are allocated 1 unit of
  cash. This is your starting Net Asset Value (NAV). If your NAV drops
  to 0, your episode is over and you lose. If your NAV hits 2.0, then
  you win.

  '''
  metadata = {'render.modes': ['human']}

  def __init__(self,**kwargs):
    for key, value in kwargs.items():
      if key == 'is_train':
        train = value

    self.minutes = DEFAULT_MINUTE_WINDOW
    self.src = EnvSrc(minutes=self.minutes, train= train)
    self.sim = TradingSim(steps=self.minutes)

    self.action_space = spaces.Discrete(3)
    self.observation_space= spaces.Box( self.src.min_values,
                                        self.src.max_values,
                                        dtype=np.float32)
    self._reset()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def _step(self, action):
    assert self.action_space.contains(action), '%r (%s) invalid'%(action, type(action))
    observation, done, ret = self.src._step()

    yret = ret 
    reward, info = self.sim._step( action, yret )
      
    #info = { 'pnl': steppnl, 'nav':self.nav, 'costs':costs }
    return observation, reward, done, info
  
  def _reset(self):
    self.src.reset()
    self.sim.reset()
    return self.src._step()[0]    

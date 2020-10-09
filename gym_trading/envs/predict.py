import os
import time
import tensorflow as tf
import datetime as dt
import gym
import policy_gradient
from time import gmtime, strftime

from Config import Config

conf = Config()

def generate_predictions(ticker):
    
    global pg

    # Initialize environment and make prediction
    env = gym.make(conf.env_name)
    env = env.unwrapped
    prediction = pg.test_model(env)

    if prediction == 0:
        pred_str = 'BUY (0)'
    elif prediction == 1:
        pred_str = 'FLAT (1)'
    elif prediction == 2:
        pred_str = 'SELL (2)'

    print('---------Predicted action is:')
    print(pred_str)
        
    time = int(dt.datetime.now().strftime('%s')) # in millisec: * 1000 
    outputString = str(prediction) + ' ' + str(time)
    
    file = open(conf.OUTPUT_PREDICT_PATH + conf.TICKER + '_prediction.txt', 'w')    
    file.write(outputString)  
    file.close()
        
if __name__ == '__main__':

    # Disable tensorflow compilation warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    # create the tf session
    sess = tf.compat.v1.InteractiveSession()

    # create policygradient
    pg = policy_gradient.PolicyGradient(sess, obs_dim=conf.observation_dimension, num_actions=conf.number_of_actions, learning_rate=conf.first_lr )
            
    generate_predictions(conf.TICKER)
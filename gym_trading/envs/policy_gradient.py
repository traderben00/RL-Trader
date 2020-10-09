import numpy as np
import gym
import tensorflow as tf
from memory_profiler import profile
from tensorflow.python.summary.writer.writer import FileWriter
import pdb
import logging
import inquirer
import os.path
import sys
import gc
import pandas as pd
import gym_trading
from time import gmtime, strftime
import time
from datetime import datetime
import train
import trading_env as te


from Config import Config

#np.random.seed(42)
#tf.random.set_seed(42)


conf = Config()

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.',__name__)

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')

simnet_dataframe_collection = pd.DataFrame(columns=['episode','simnet'])

current_action = 'FLAT'
buy_sell_num = 0

class PolicyGradient(object) :
    @profile
    def __init__(self,
                 sess,                          # tensorflow session
                 obs_dim,                       # observation shape
                 num_actions,                   # number of possible actions
                 neurons_per_dim=32,            # hidden layer will have obs_dim * neurons_per_dim neurons
                 learning_rate=conf.first_lr,   # learning rate
                 gamma = conf.gamma,            # reward discounting 
                 decay = conf.decay             # gradient decay rate
                 ):
                 
        self._sess = sess
        self._gamma = gamma
        self._tf_model = {}
        self._num_actions = num_actions
        hidden_neurons = obs_dim * neurons_per_dim
        with tf.compat.v1.variable_scope('layer_one',reuse=False):
            L1 = tf.truncated_normal_initializer(mean=0,
                                                 stddev=1./np.sqrt(obs_dim),
                                                 dtype=tf.float32)
            self._tf_model['W1'] = tf.compat.v1.get_variable('W1',
                                                   [obs_dim, hidden_neurons],
                                                   initializer=L1)
        with tf.compat.v1.variable_scope('layer_two',reuse=False):
            L2 = tf.truncated_normal_initializer(mean=0,
                                                 stddev=1./np.sqrt(hidden_neurons),
                                                 dtype=tf.float32)
            self._tf_model['W2'] = tf.compat.v1.get_variable('W2',
                                                   [hidden_neurons,num_actions],
                                                   initializer=L2)
       
        # tf placeholders
        self._tf_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, obs_dim],name='tf_x')
        self._tf_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_actions],name='tf_y')
        self._tf_epr = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,1], name='tf_epr')

        # tf reward processing (need tf_discounted_epr for policy gradient wizardry)
        self._tf_discounted_epr = self.tf_discount_rewards(self._tf_epr)
        self._tf_mean, self._tf_variance= tf.nn.moments(self._tf_discounted_epr, [0], 
                                                        shift=None, name='reward_moments')
        self._tf_discounted_epr -= self._tf_mean
        self._tf_discounted_epr /= tf.sqrt(self._tf_variance + 1e-6)

        self._saver = tf.compat.v1.train.Saver()


        # tf optimizer op
        self._tf_aprob = self.tf_policy_forward(self._tf_x)
        loss = tf.nn.l2_loss(self._tf_y - self._tf_aprob) # this gradient encourages the actions taken
        tf.compat.v1.summary.scalar('loss', loss)

        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay)

        tf_grads = optimizer.compute_gradients(loss, var_list=tf.compat.v1.trainable_variables(), 
                                               grad_loss=self._tf_discounted_epr)
        self._train_op = optimizer.apply_gradients(tf_grads)
    @profile
    def tf_discount_rewards(self, tf_r): #tf_r ~ [game_steps,1]
        discount_f = lambda a, v: a*self._gamma + v;
        tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
        tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
        return tf_discounted_r
    @profile
    def tf_policy_forward(self, x): #x ~ [1,D]
        h = tf.matmul(x, self._tf_model['W1'])
        tf.compat.v1.summary.histogram('weights', h)
        h = tf.nn.relu(h)
        tf.compat.v1.summary.histogram('activations', h)
        logp = tf.matmul(h, self._tf_model['W2'])
        tf.compat.v1.summary.histogram('weights', logp)
        p = tf.nn.softmax(logp)
        tf.compat.v1.summary.histogram('activations', p)
        return p

    @profile
    def train_model(self, env, episodes=1000, 
                    load_model = True,  # load model from checkpoint if available:?
                    model_dir = conf.MODEL_DIR, log_freq=50, thread_cond=None ) :

        global simnet_dataframe_collection
        global current_action
        global buy_sell_num

        min_evaluation_size = int(conf.NO_OF_EPISODES / 10)

        
        # Check whether there is already any model
        for dirpath, dirnames, files in os.walk(model_dir):
            if files:
                print(dirpath, '\nfolder has files\n')
                print('The ' + dirpath + ' folder contains one or more files which can cause confuse during the training.\n')
                first_ans = 'Start training without action (YOUR EXISTING MODELS MAY BE OVERWRITTEN!)'
                sec_ans = 'Clear all files in the folder'
                third_ans = 'Exit program without action'
                questions = [ inquirer.List('action',
                        message='Which action do you want to take?',
                        choices=[first_ans, sec_ans, third_ans], ),]
                answers = inquirer.prompt(questions)
                if sec_ans in answers.values():
                    for the_file in os.listdir(model_dir):
                        file_path = os.path.join(model_dir, the_file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                        except Exception as e:
                            print(e)
                elif third_ans in answers.values():
                    sys.exit(0)
                break
            if not files:
                print(dirpath, '\nfolder is empty') 
        

        # initialize variables and load model
        writer = tf.compat.v1.summary.FileWriter('tmp/' + conf.TICKER + '/' + now)
        writer.add_graph(self._sess.graph)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        if load_model:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            print(tf.train.latest_checkpoint(model_dir))
            if ckpt and ckpt.model_checkpoint_path:
                #savr = tf.compat.v1.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
                out = self._saver.restore(self._sess, ckpt.model_checkpoint_path)
                print('Model restored from ',ckpt.model_checkpoint_path)
            else:
                print('No checkpoint found at: ',model_dir)

        actions_tofile = pd.DataFrame(columns=['buy', 'sell', 'step', 'episode', 'price'])

        episode = 0
        observation = env.reset()
        xs,rs,ys = [],[],[]    # environment info
        running_reward = 0    
        reward_sum = 0
        steps = 0
        simrors = np.zeros(episodes)
        mktrors = np.zeros(episodes)
        victory = False
        
        # === Training starts here ===
        while episode < episodes and not victory:
            if steps == 0:
                start_time_episode = time.time()
            
            # stochastically sample a policy from the network
            x = observation
            feed = {self._tf_x: np.reshape(x, (1,-1))}
            aprob = self._sess.run(self._tf_aprob,feed)
            aprob = aprob[0,:]            
                        
            # For example there is a long position then the next action must be FLAT or SELL and vice versa
            if current_action == 'BUY':
                aprob[2] = 0
                aprob[0] = aprob[2] / 2 + aprob[0]
                aprob[1] = aprob[2] / 2 + aprob[1]
            elif current_action == 'SELL':
                aprob[0] = 0
                aprob[2] = aprob[0] / 2 + aprob[2]
                aprob[1] = aprob[0] / 2 + aprob[1]
   
            
            aprob /= aprob.sum()
            action = np.random.choice(self._num_actions, p=aprob)


            # Write actions to file (for the plot)
            if action == 0:
                actions_tofile = actions_tofile.append({'sell' : observation[0], 'price' : observation[0], 'step' : steps, 'episode' : int(episode)}, ignore_index=True)
                current_action = 'SELL'
                buy_sell_num = buy_sell_num + 1
            elif action == 2:
                actions_tofile = actions_tofile.append({'buy' : observation[0], 'price' : observation[0], 'step' : steps, 'episode' : int(episode)}, ignore_index=True)
                current_action = 'BUY'
                buy_sell_num = buy_sell_num + 1
            else:
                actions_tofile = actions_tofile.append({'price' : observation[0], 'step' : steps, 'episode' : int(episode)}, ignore_index=True)

            label = np.zeros_like(aprob) ; label[action] = 1 # make a training 'label'
            
            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            reward_sum += reward

            # record game history collected in arrays
            xs.append(x)        # 'X' is the actual step's observation (price)
            ys.append(label)    # 'label' is an array which looks like [0, 1, 0]. The location of the number '1' shows what action was taken on this step. This exaple: FLAT
            rs.append(reward)   # 'reward' is a value which shows how much reward the system gets in this step
            steps += 1
            if done:
                print('--------LAST STEP IN EPISODE: ' + str(episode))
                # This reinforces the system to take enough actions
                if buy_sell_num < conf.num_of_rows_read * 0.01:
                    reward_sum = -1
                
                print('Reward sum = ',reward_sum)
                running_reward = running_reward * 0.99 + reward_sum * 0.01
                
                # vstack: Stack arrays in sequence vertically (row wise)
                epx = np.vstack(xs)
                epr = np.vstack(rs)
                epy = np.vstack(ys)

                xs,rs,ys = [],[],[] # reset game history
                df = env.sim.to_df()
                #pdb.set_trace()

                simrors[episode]=df.bod_nav.values[-1]-1 # compound returns
                mktrors[episode]=df.mkt_nav.values[-1]-1
                
                summ = tf.compat.v1.summary.merge_all()

                start_time_update_model = time.time()
                    
                feed = {self._tf_x: epx, self._tf_epr: epr, self._tf_y: epy}

                summary,_ = self._sess.run([summ,self._train_op],feed) # parameter update with TensorBoard
                writer.add_summary(summary, episode)

                end_time_update_model = time.time()
                elapsed_time_update_model = end_time_update_model - start_time_update_model
                print('Updating the model took: ' +str(elapsed_time_update_model)+ ' seconds\n')

                if episode % conf.LOG_FREQ == 0:
                    start_time_save_model = time.time()
                    save_path = self._saver.save(self._sess, model_dir + 'model.ckpt',
                                                 global_step=episode+1)
                    end_time_save_model = time.time()
                    elapsed_time_save_model = end_time_save_model - start_time_save_model
                    print('Model saved in file: ' +str(save_path) + ', Took: ' +str(elapsed_time_save_model)+ ' seconds\n')                   
                    
                    if not os.path.exists(conf.PLOT_PATH):
                        os.makedirs(conf.PLOT_PATH)
                    if not os.path.exists(conf.PLOT_PATH + conf.simnet):
                        os.makedirs(conf.PLOT_PATH + conf.simnet)

                    simnet_dataframe_collection = simnet_dataframe_collection.append({'episode' : int(episode) , 'simnet' : simrors[episode] - mktrors[episode]}, ignore_index=True)
                    
                    simnet_dataframe_collection.to_csv(conf.PLOT_PATH + conf.simnet + conf.TICKER + '_simnet' + conf.csv_file, sep=',', mode='w', index=False)

                    actions_tofile.to_csv(conf.PLOT_PATH + 'actions/' + conf.TICKER + conf.actions_path_extension, sep=',')

                    outputString = 'episode #%6d, mean reward: %8.4f, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f' % (episode,
                             running_reward, simrors[episode],mktrors[episode], simrors[episode]-mktrors[episode])

                    log.info('episode #%6d, mean reward: %8.9f, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', episode,
                             running_reward, simrors[episode],mktrors[episode], simrors[episode]-mktrors[episode])

                    stat_path = conf.PLOT_PATH + 'Training_Stats.txt'
                    if os.path.exists(stat_path):
                        os.remove(stat_path)
                    file = open(stat_path, 'a')    
                    file.write(outputString)  
                    file.close()

                if episode > min_evaluation_size:
                    vict = pd.DataFrame( { 'sim': simrors[episode-min_evaluation_size:episode],
                                           'mkt': mktrors[episode-min_evaluation_size:episode] } )
                    vict['net'] = vict.sim - vict.mkt
                    vict_net_str = ', VICT_NET mean is: ' + str(vict.net.mean()) + '\n'
                    print(vict_net_str)
                    if vict.net.mean() > 0.00:
                        victory = True
                        log.info('Congratulations, Warren Buffett! One of your life achievements unlocked. You can progress to another greater stage of your life.')
                        print(str(simrors - mktrors))          

                if episode < conf.first_lr_change: self._train_op.learning_rate = conf.first_lr
                if episode > conf.first_lr_change and episode < conf.sec_lr_change: self._train_op.learning_rate = conf.sec_lr
                if episode > conf.sec_lr_change: self._train_op.learning_rate = conf.third_lr
                    
                actions_tofile.drop(actions_tofile.index, inplace=True) # We store the actual episode's actions only
                end_time_episode = time.time()
                elapsed_time_episode = end_time_episode - start_time_episode
                print('Training episode: ' +str(episode)+ ' took: ' +str(elapsed_time_episode)+' seconds\n')
                episode += 1
                buy_sell_num = 0
                observation = env.reset()
                reward_sum = 0
                steps = 0
                gc.collect()
                
    def test_model(self, env,
                    model_dir = conf.MODEL_DIR ) :

        # initialize variables and load model
        init_op = tf.compat.v1.global_variables_initializer()
        self._sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        print(tf.train.latest_checkpoint(model_dir))
        if ckpt and ckpt.model_checkpoint_path:
            savr = tf.compat.v1.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
            out = savr.restore(self._sess, ckpt.model_checkpoint_path)
            print('Model restored from ',ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found at: ',model_dir)

        observation = env.reset()
        xs,rs,ys = [],[],[]    # environment info
        steps = 0
        
        # stochastically sample a policy from the network
        x = observation
        feed = {self._tf_x: np.reshape(x, (1,-1))}
        aprob = self._sess.run(self._tf_aprob,feed)
        aprob = aprob[0,:] # we live in a batched world :/

        action = np.random.choice(self._num_actions, p=aprob)
        return action
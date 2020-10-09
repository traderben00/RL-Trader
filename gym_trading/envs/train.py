import gym
import trading_env as te
import tensorflow as tf
import policy_gradient
import gc

from Config import Config

if __name__ == '__main__':

	conf = Config()

	env = gym.make(conf.env_name, is_train=conf.IS_TRAIN)
	env = env.unwrapped

	gc.collect()
	tf.compat.v1.reset_default_graph()
	
	# create the tf session
	sess = tf.compat.v1.InteractiveSession()
	sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
	
	# create policy_gradient
	pg = policy_gradient.PolicyGradient(sess, obs_dim=conf.observation_dimension, num_actions=conf.number_of_actions, learning_rate=conf.first_lr )

	print('The training progress is going to last for episode:')
	print(str(conf.NO_OF_EPISODES))

	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)

	sf = pg.train_model( env,episodes=conf.NO_OF_EPISODES, model_dir= conf.MODEL_DIR, log_freq=conf.LOG_FREQ )
	sf['net'] = sf.simrors - sf.mktrors
	
	print(sf['net'])
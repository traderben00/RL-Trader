import os
import tempfile

class Config:
		
	IS_TRAIN = True # Set whether you want to Train (True) or Predict (False)
	TICKER = 'EURUSD'

	num_of_rows_read = 1000 # If set 0 then all the rows will be read

	# Set MySQL inputs if True
	IS_MYSQL = False

	MYSQL_USER = 'Write your user name'
	MYSQL_PASSWORD = 'Write your password'
	MYSQL_HOST = 'Write the IP address of the MySQL'
	MYSQL_DATABASE = 'Write the name of the database where your dataset can be found'
	MYSQL_PORT = 0 # your mysql port number
	MYSQL_HOST_PORT = MYSQL_HOST +':'+ str(MYSQL_PORT)

	# Env params
	env_name = 'trading-v0'
	number_of_actions = 3 # Short (0), Flat (1), Long (2)
	observation_dimension = 27 # Number of Features (you have to change it unless you have 27 features of your dataset)
	gamma = 0.9
	decay = 0.9
	execution_penalty = 0.0001 #0.001
	timestep_penalty = 0.0001

	# Set the adaptive learning rate
		# Changing points in episode number
	first_lr_change = 500 
	sec_lr_change = 60000
	third_lr_change = 80000

		# Learning rate values
	first_lr = 1e-4
	sec_lr = 1e-3
	third_lr = 1e-3

	# Training params
	NO_OF_EPISODES = 10000
	LOG_FREQ = 10
	LOGDIR = '/tensorboard/'	# Log path for the tensorboard
	MODEL_DIR = 'model/'	# Path for saving models
	

	# Extensions
	csv_file = '.csv'
	input_predict_extension = '_input_predict' + csv_file
	simnet = 'simnet/'
	simnet_path_extension = '_simnet.csv'
	actions_path_extension = '_actions.csv'

	# Path sources
	INPUT_PREDICT_DATA_PATH = os.path.join('datasets', 'input_predict/')
	TRAINING_DATA_PATH = os.path.join('datasets', 'training/')
	PLOT_PATH = 'plot/'
	OUTPUT_PREDICT_PATH = os.path.join('datasets', 'output_predict/')
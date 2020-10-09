# RL-Trader

## Introduction
This Reinforcement Learning agent is using Policy-Gradient method to find buy and sell points on the financial markets.
There are example datasets on which you can run the program.


## How To Use
1. Check your missing dependencies below and install them.
2. Setup gym: Go to the root folder of the repo and use this command:
```sh
$ pip install -e .
```
3. Check the 'Config.py' file and set the desired parameters (you can let them on default to test).
4. Run the 'train.py' to train your model.
Datasets located in your MySQL can also be used. To use MySQL, set the parameters in the 'Config.py' file and install the required dependencies.
Local training example dataset is located in the folder 'datasets/training'.
5. Having trained the model, Run the 'predict.py' file to make a prediction. An example data to be predicted can be found in the folder 'datasets/input_predict'.
6. See your actual prediction result file in the folder 'datasets/output_predict'.
7. Make plots to see how your agent performs. Run the 'make_plot.py' and see the plots about the returns and the actions that the agent took.

After the training has finished you can analyze the process with tensorboard. (tensorboard --logdir tmp/your_ticker_name/)

#### Other convenience features:
- Upload datasets to MySQL:
Set the desired parameters in the file 'fill_mysql_database.py', then run it to upload your dataset.
- Check MySQL tables:
Print out any MySQL table. Use the file 'check_mysql_table.py'.
- Convert raw datasets to the particular dataformat that the program uses:
The converter can be found in the folder 'data_convert'. Open up the file 'data_convert.py' and edit the 'data_path' variable implicitly. The format of your dataset has to be exactly the same as the example file called 'EURUSD_example_100000.csv'. You can test the converter on this dataset. The training uses the dataset located in the folder 'datasets/training'.

## Used Versions 

The program was tested under these dependencies:

Dependencies | Version number
------------ | -------------
Python | 3.6.5.final.0
python-bits | 64
OS | Linux
OS-release | 4.15.0-54-generic
machine | x86_64
processor | x86_64
Pip | 20.0.2
Tensorflow | 1.14.0
Pandas | 0.24.2
Numpy | 1.16.4
Scipy | 1.3.0
Matplotlib | 2.0.2
Gym | 0.10.11
SQLAlchemy | 1.3.4


## Results

This agent was trained for only 160 episodes to give an insight, example.
The model, tensorboard files, plots and the result pictures can also be found in the folders.


On this figure, the taken (buy and sell) actions can be seen.
![Actions that the agent took](/gym_trading/envs/result/actions.png)

This figure is telling us the calculated returns in the course of the learning process.
![Net return](/gym_trading/envs/result/simnet.png)

## Contact
Any suggestion, help, contribution would be highly appreciated.


Bence Szabo

LinkedIn:
https://www.linkedin.com/in/ben-szabo/

E-mail:
traderben00@gmail.com

## License

Creative Commons

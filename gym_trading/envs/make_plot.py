import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from Config import Config

conf = Config()

instrument = conf.TICKER 

# Plot simnet

sf_net = pd.read_csv(conf.PLOT_PATH + conf.simnet + instrument + '_simnet' + conf.csv_file, index_col=False)

print('Sim_net to plot')
print(sf_net)

plt.plot(sf_net['episode'], sf_net['simnet'], color='green', label='true', linewidth=1)
plt.title('Simnet')
plt.ylabel('Simnet')
plt.xlabel('Episode')

plt.show()
plt.gcf().clear()

# Plot actions
actions = pd.read_csv(conf.PLOT_PATH + 'actions/' + instrument + conf.actions_path_extension, index_col=False)

print('Actions to plot:')
print(actions)

plt.plot(actions['step'], actions['buy'], 'g^')
plt.plot(actions['step'], actions['sell'], 'rv')
plt.plot(actions['step'], actions['price'], 'k-', linewidth=0.7)

plt.title('Actions for episode: ' + str(int(actions['episode'][1])))
plt.ylabel('Price')
plt.xlabel('Step')

plt.show()
plt.gcf().clear()



# Trading stats
steps_number = actions['step'].count()
print('All steps:')
print(steps_number)
buys_number = actions['buy'].count()
sells_number = actions['sell'].count()
trades_number = buys_number + sells_number
buy_ratio = buys_number / trades_number * 100
sell_ratio = sells_number / trades_number * 100
average_profit = sf_net['simnet'].iloc[-1] / trades_number
print('\nAll actions:')
print(trades_number)
print('\nBuy actions:')
print('(' + str(buy_ratio.round(2)) + ' %)')
print('\nSell actions:')
print('(' + str(sell_ratio.round(2)) + ' %)')
print('\nAverage simnet:')
print(average_profit)
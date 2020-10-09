from gym.envs.registration import register

register(
    id='trading-v0',
    entry_point='gym_trading.envs:TradingEnv',
    timestep_limit=None,
    kwargs={'is_train': True}
)
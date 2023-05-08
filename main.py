from Backtesting import backtesting_base as bt
from mt5_global import settings
from agent import trading_agent as TradingAgent
from environment import finance_env as finanEnv
from Trade_platform import meta_trader
from mt5_actions.authorize import login

symbol = settings.symbol
lot_size = 0.01
# symbol for yahoo finance data eurusd=x
symbol_y = settings.symbol
features = [symbol_y, 'r', 's', 'm', 'v']
a = 0
b = 2000
c = 500




def trade():
    if not login():
        print('login failed')
        return
    else:
        learn_env = finanEnv.Finance(symbol_y, features, window=10, lags=6,
                                     leverage=100, min_performance=0.85,
                                     start=a, end=a + b, mu=None, std=None)


        valid_env = finanEnv.Finance(symbol_y, features, window=learn_env.window,
                                    lags=learn_env.lags, leverage=learn_env.leverage,
                                    min_performance=learn_env.min_performance,
                                    start=a + b, end=a + b + c,
                                    mu=learn_env.mu, std=learn_env.std)


        env = learn_env
        agent = TradingAgent.TradingBot(24, 0.0001, env, valid_env=valid_env)
        episodes = 40
        agent.learn(episodes)
        agent.plot_treward()
        agent.plot_performance()
        mt5_trade = meta_trader.Mt5TradingBot(
            agent=agent, granularity=15,units=0.01, lot_size=lot_size)
    while True:
        mt5_trade.run()


if __name__ == '__main__':
    trade()

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from Backtesting_with_risk_measure import backtest_w_risk_measure as bb_w_risk_measure
from agent import trading_agent as TradingAgent
from environment import finance_env as finanEnv
from mt5_actions.authorize import login
from mt5_global import settings




import MetaTrader5 as mt5

# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()


class TBBacktester(bb_w_risk_measure.BacktestingBaseRM):
    def _reshape(self, state):
        ''' Helper method to reshape state objects.
        '''
        return np.reshape(state, [1, self.env.lags, self.env.n_features])


    def backtest_strategy(self, sl=None, tsl=None, tp=None,
                        wait=5, guarantee=False):

        ''' Event-based backtesting of the trading bot's performance.
        Incl. stop loss, trailing stop loss and take profit.
        '''
        self.units = 0
        self.position = 0
        self.trades = 0
        self.sl = sl
        self.tsl = tsl
        self.tp = tp
        self.wait = 0
        self.current_balance = self.initial_amount
        self.net_wealths = list()
        for bar in range(self.env.lags, len(self.env.data)):
            self.wait = max(0, self.wait - 1)
            date, price = self.get_date_price(bar)
            if self.trades == 0:
                print(50 * '=')
                print(f'{date} | *** START BACKTEST ***')
                self.print_balance(bar)
                print(50 * '=')
            # stop loss order
            if sl is not None and self.position != 0:
                rc = (price - self.entry_price) / self.entry_price
                if self.position == 1 and rc < -self.sl:
                    print(50 * '-')
                    if guarantee:
                        price = self.entry_price * (1 - self.sl)
                        print(f'*** STOP LOSS (LONG | {-self.sl:.4f}) ***')
                    else:
                        print(f'*** STOP LOSS (LONG | {rc:.4f}) ***')
                    self.place_sell_order(bar, units=self.units, gprice=price)
                    self.wait = wait
                    self.position = 0
                elif self.position == -1 and rc > self.sl:
                    print(50 * '-')
                    if guarantee:
                        price = self.entry_price * (1 + self.sl)
                        print(f'*** STOP LOSS (SHORT | -{self.sl:.4f}) ***')
                    else:
                        print(f'*** STOP LOSS (SHORT | -{rc:.4f}) ***')
                    self.place_buy_order(bar, units=-self.units, gprice=price)
                    self.wait = wait
                    self.position = 0
               
            # trailing stop loss order
            if tsl is not None and self.position != 0:
                self.max_price = max(self.max_price, price)
                self.min_price = min(self.min_price, price)
                rc_1 = (price - self.max_price) / self.entry_price
                rc_2 = (self.min_price - price) / self.entry_price
                if self.position == 1 and rc_1 < -self.tsl:
                    print(50 * '-')
                    print(f'*** TRAILING SL (LONG | {rc_1:.4f}) ***')
                    self.place_sell_order(bar, units=self.units)
                    self.wait = wait
                    self.position = 0
                elif self.position == -1 and rc_2 < -self.tsl:
                    print(50 * '-')
                    print(f'*** TRAILING SL (SHORT | {rc_2:.4f}) ***')
                    self.place_buy_order(bar, units=-self.units)
                    self.wait = wait
                    self.position = 0
            # take profit order
            if tp is not None and self.position != 0:
                rc = (price - self.entry_price) / self.entry_price
                if self.position == 1 and rc > self.tp:
                    print(50 * '-')
                    if guarantee:
                        price = self.entry_price * (1 + self.tp)
                        print(f'*** TAKE PROFIT (LONG | {self.tp:.4f}) ***')
                    else:
                        print(f'*** TAKE PROFIT (LONG | {rc:.4f}) ***')
                        self.place_sell_order(bar, units=self.units, gprice=price)
                        self.wait = wait
                        self.position = 0
                elif self.position == -1 and rc < -self.tp:
                    print(50 * '-')
                    if guarantee:
                        price = self.entry_price * (1 - self.tp)
                        print(f'*** TAKE PROFIT (SHORT | {self.tp:.4f}) ***')
                    else:
                        print(f'*** TAKE PROFIT (SHORT | {-rc:.4f}) ***')
                    self.place_buy_order(bar, units=-self.units, gprice=price)
                    self.wait = wait
                    self.position = 0
            state = self.env.get_state(bar)
            action = np.argmax(self.model.predict(
                self._reshape(state.values),verbose=False)[0, 0])
            position = 1 if action == 1 else -1
            if self.position in [0, -1] and position == 1 and self.wait == 0:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOING LONG ***')
                if self.position == -1:
                    self.place_buy_order(bar - 1, units=-self.units)
                self.place_buy_order(bar - 1, amount=self.current_balance)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = 1
            elif self.position in [0, 1] and position == -1 and self.wait == 0:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOING SHORT ***')
                if self.position == 1:
                    self.place_sell_order(bar - 1, units=self.units)
                self.place_sell_order(bar - 1, amount=self.current_balance)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = -1
        series = pd.Series(
            {'date': date, 'net_wealth': self.calculate_net_wealth(price)}, name = bar)
        self.net_wealths.append(series)
        self.net_wealths = pd.DataFrame(self.net_wealths,
                                        columns=['date', 'net_wealth'])
        self.net_wealths.set_index('date', inplace=True)
        self.net_wealths.index = pd.DatetimeIndex(self.net_wealths.index)
        self.close_out(bar)


symbol = settings.symbol
# symbol for yahoo finance data eurusd=x
symbol_y = settings.symbol
features = [symbol_y, 'r', 's', 'm', 'v']
a = 0
b = 2000
c = 500



def test():
    if not login():
        print('Login failed.')
        return
    print('Login successful.')
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
    episodes = 60

    agent.learn(episodes)
    agent.plot_treward()
    agent.plot_performance()

    plt.show()


    tb = TBBacktester(env, agent.model, 10000,
                    0.0, 0, verbose=False)

    tb.backtest_strategy(sl=None, tsl=None, tp=None, wait=5)


    tb.backtest_strategy(sl=0.0175, tsl=None, tp=None,
                        wait=5, guarantee=False)


    tb.backtest_strategy(sl=0.017, tsl=None, tp=None,
                        wait=5, guarantee=True)

    tb.backtest_strategy(sl=None, tsl=0.015,
                        tp=None, wait=5)

    tb.backtest_strategy(sl=None, tsl=None, tp=0.015,
                        wait=5, guarantee=False)


    tb.backtest_strategy(sl=None, tsl=None, tp=0.015,
                        wait=5, guarantee=True)

    tb.backtest_strategy(sl=0.015, tsl=None,
                        tp=0.0185, wait=5)


    tb.backtest_strategy(sl=None, tsl=0.02,
                        tp=0.02, wait=5)





if __name__ == '__main__':
    test()
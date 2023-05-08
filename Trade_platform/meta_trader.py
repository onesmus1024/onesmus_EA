import pandas as pd
import numpy as np
import pprint
import datetime as dt
import MetaTrader5 as mt5
from mt5_global import settings
from mt5_actions import order

timeframe = settings.timeframe

symbol = settings.symbol
Model_type = settings.Model_type



class Mt5TradingBot:


    def __init__(self,agent, granularity, units,lot_size,
                verbose=True):
        self.agent = agent
        self.symbol = self.agent.learn_env.symbol
        self.env = agent.learn_env
        self.window = self.env.window
        if granularity is None:
            self.granularity = agent.learn_env.granularity
        else:
            self.granularity = granularity
        self.units = units
        self.trades = 0
        self.position = 0
        self.tick_data = pd.DataFrame()
        self.min_length = (self.agent.learn_env.window +
                        self.agent.learn_env.lags)
        self.pl = list()
        self.verbose = verbose
        self.data= self._get_data()
        self._prepare_data()

    def _get_data(self):

        # self.raw = pd.read_csv(self.url, index_col=0,
        #                        parse_dates=True).dropna()
        # end should be now
        self.raw = pd.DataFrame(mt5.copy_rates_from_pos(
            settings.symbol, settings.timeframe, 0, 2000))
        self.raw[self.symbol] = self.raw['close']
        #drop all columns except self.symbol
        self.raw.drop(self.raw.columns.difference([self.symbol]), 1, inplace=True)
        
        
        return self.raw


    def _prepare_data(self):

        ''' Prepares the (lagged) features data.
        '''
        self.data = pd.DataFrame(self.raw[self.symbol])
        self.data['r'] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace=True)
        self.data['s'] = self.data[self.symbol].rolling(self.window).mean()
        self.data['m'] = self.data['r'].rolling(self.window).mean()
        self.data['v'] = self.data['r'].rolling(self.window).std()
        self.data.dropna(inplace=True)
        self.data_ = (self.data - self.env.mu) / self.env.std


    def _resample_data(self):

        ''' Resamples the data to the trading bar length.
        '''
        self.data = self.tick_data.resample(self.granularity,
                                            label='right').last().ffill().iloc[:-1]
        self.data = pd.DataFrame(self.data['mid'])
        self.data.columns = [self.symbol,]
        self.data.index = self.data.index.tz_localize(None)


    def _get_state(self):

        ''' Returns the (current) state of the financial market.
        '''
        state = self.data_[self.env.features].iloc[-self.env.lags:]
        return np.reshape(state.values, [1, self.env.lags, self.env.n_features])


    def report_trade(self, time, side, order):

        ''' Reports trades and order details.
        '''
        self.trades += 1
        pl = float(order['pl'])
        self.pl.append(pl)
        cpl = sum(self.pl)
        print('\n' + 71 * '=')
        print(f'{time} | *** GOING {side} ({self.trades}) ***')
        print(f'{time} | PROFIT/LOSS={pl:.2f} | CUMULATIVE={cpl:.2f}')
        print(71 * '=')
        if self.verbose:
            pprint(order)
            print(71 * '=')

    def get_sl(self):
        data = self._get_data()

        data['min'] = data[self.symbol].rolling(self.window).min()
        data['max'] = data[self.symbol].rolling(self.window).max()
        data['mami'] = data['max'] - data['min']
        data['mac'] = abs(data['max'] - data[self.symbol].shift(1))
        data['mic'] = abs(data['min'] - data[self.symbol].shift(1))
        data['atr'] = np.maximum(data['mami'], data['mac'])
        data['atr'] = np.maximum(data['atr'], data['mic'])

        ATR = data['atr'].iloc[-1]

        return ATR


    def run(self):

        ''' Contains the main trading logic.
        '''
        
        latest_candle = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 1)[0]
        if latest_candle["time"] != mt5.copy_rates_from_pos(self.symbol, timeframe, 1, 1)[0]["time"]:
            self.min_length += 1
            self._get_data()
            self._prepare_data()
            state = self._get_state()
            prediction = np.argmax(self.agent.model.predict(state)[0, 0])
            position = 1 if prediction == 1 else -1
            if self.position in [0, -1] and position == 1:
                stop_loss = mt5.symbol_info_tick(self.symbol).ask - self.get_sl()
                order.buy_order(self.symbol,stop_loss)
                self.position = 1
            elif self.position in [0, 1] and position == -1:
               # Place a short trade (sell)
                stop_loss = mt5.symbol_info_tick(self.symbol).bid + self.get_sl()
                order.sell_order(self.symbol,stop_loss)
                self.position = -1
        else:
            print("No new candle yet")
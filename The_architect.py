import pandas as pd
import numpy as np
import talib
from hurst import compute_Hc
from scipy.stats import entropy as scipy_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from enum import Enum
from itertools import product


class TradeType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    REVERSE_LONG = "REVERSE_LONG"
    REVERSE_SHORT = "REVERSE_SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


class Strategy:
    def run(self, data: pd.DataFrame, vol_thresholds=None, estimator_thresholds=None,
            base_take_profit_multiplier=11, base_stop_loss_multiplier=11) -> pd.DataFrame:
        class Indicators:
            def __init__(self, data):
                self.data = data
                self.close = data['close'].values

            def hurst_exponent(self, window_data):
                try:
                    H, _, _ = compute_Hc(
                        window_data, kind='price', simplified=True)
                    return H if not np.isnan(H) else 0.5
                except:
                    return 0.5

            def fdi(self, window_data):
                returns = np.log(window_data[1:]) - np.log(window_data[:-1])
                fdi = 2 - self.hurst_exponent(window_data)
                return fdi if not np.isnan(fdi) else 1.5

            def bollinger_bands(self, prices, period=20):
                upper, middle, lower = talib.BBANDS(prices, timeperiod=period)
                return (
                    pd.Series(upper).fillna(method='ffill').fillna(prices[0]),
                    pd.Series(middle).fillna(method='ffill').fillna(prices[0]),
                    pd.Series(lower).fillna(method='ffill').fillna(prices[0])
                )

            def entropy(self, window_data, window=20):
                returns = pd.Series(window_data).pct_change().dropna()
                if len(returns) < window:
                    return 2.16
                hist, bin_edges = np.histogram(
                    returns[-window:], bins=20, density=True)
                ent = scipy_entropy(hist + 1e-10)
                return ent if not np.isnan(ent) else 2.16

            def macd(self, prices, fastperiod=50, slowperiod=100, signalperiod=20):
                macd, signal, _ = talib.MACD(
                    prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                return pd.Series(macd).fillna(method='ffill').fillna(0)

            def sma(self, prices, period=28):
                sma = talib.SMA(prices, timeperiod=period)
                return pd.Series(sma).fillna(method='ffill').fillna(prices[0])

            def ema(self, prices, period=28):
                ema = talib.EMA(prices, timeperiod=period)
                return pd.Series(ema).fillna(method='ffill').fillna(prices[0])

            def rsi(self, prices, period=21):
                rsi = talib.RSI(prices, timeperiod=period)
                return pd.Series(rsi).fillna(method='ffill').fillna(50)

            def obv(self, prices, volume):
                obv = talib.OBV(prices, volume)
                return pd.Series(obv).fillna(method='ffill').fillna(0)

            def stochastic(self, high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
                slowk, slowd = talib.STOCH(
                    high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
                return pd.Series(slowk).fillna(method='ffill').fillna(50)

            def vwap(self, high, low, close, volume):
                typical_price = (high + low + close) / 3
                vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
                return pd.Series(vwap).fillna(method='ffill').fillna(close[0])

            def velocity(self, param_series):
                return np.diff(param_series) if len(param_series) > 1 else np.array([0])

        class TradingStrategy:
            def __init__(self, data, vol_thresholds, estimator_thresholds, base_take_profit_multiplier, base_stop_loss_multiplier):
                self.data = data
                self.btc_indicators = Indicators(data)
                self.signals = pd.DataFrame(index=data.index)
                self.t_recalc = 100
                self.scaler = StandardScaler()
                self.position = 0
                self.is_rf_fitted = False
                self.btc_rf = None
                self.vol_thresholds = vol_thresholds if vol_thresholds else {
                    0.005: 0.0085, 0.01: 0.0075, 0.02: 0.010, 0.03: 0.015, float('inf'): 0.024
                }

                self.estimator_thresholds = estimator_thresholds if estimator_thresholds else {
                    0.005: 55, 0.01: 45, 0.02: 35, 0.03: 45, float('inf'): 60
                }
                self.base_take_profit_multiplier = base_take_profit_multiplier
                self.base_stop_loss_multiplier = base_stop_loss_multiplier
                self.take_profit_multipliers = {
                    0.005: self.base_take_profit_multiplier * 0.7,  # Low vol: tighter TP
                    0.01: self.base_take_profit_multiplier * 0.85,
                    0.02: self.base_take_profit_multiplier * 0.95,
                    0.03: self.base_take_profit_multiplier * 1.1,
                    # High vol: wider TP
                    float('inf'): self.base_take_profit_multiplier * 1.3
                }
                self.stop_loss_multipliers = {
                    0.005: self.base_stop_loss_multiplier * 0.75,  # Low vol: tighter SL
                    0.01: self.base_stop_loss_multiplier * 0.85,
                    0.02: self.base_stop_loss_multiplier * 0.95,
                    0.03: self.base_stop_loss_multiplier * 1,
                    # High vol: wider SL
                    float('inf'): self.base_stop_loss_multiplier * 1.2
                }
                self.atr_period = 21
                self.atr = talib.ATR(
                    self.data.get('high_btc', self.data['close']).values,
                    self.data.get('low_btc', self.data['close']).values,
                    self.data['close'].values,
                    timeperiod=self.atr_period
                )
                self.atr = pd.Series(self.atr, index=self.data.index).fillna(
                    method='ffill').fillna(0)
                self.entry_price = None
                self.stop_loss_cooldown = 0
                self.max_cooldown = 14

            def get_volatility_params(self, prices):
                if len(prices) < 2:
                    # Default position = 60
                    return 0.01, 35, self.base_take_profit_multiplier, self.base_stop_loss_multiplier, 60
                returns = pd.Series(prices).pct_change().dropna()
                vol = returns.std()
                # Map volatility to position % (60 to 100)
                vol_min, vol_max = 0.005, 0.025  # Define reasonable volatility range
                position_pct = 50 + 30 * \
                    (min(max((vol - vol_min) / (vol_max - vol_min), 0), 1))
                for vol_level in sorted(self.vol_thresholds.keys()):
                    if vol < vol_level:
                        return (
                            self.vol_thresholds[vol_level],
                            self.estimator_thresholds[vol_level],
                            self.take_profit_multipliers[vol_level],
                            self.stop_loss_multipliers[vol_level],
                            position_pct
                        )
                return (
                    self.vol_thresholds[float('inf')],
                    self.estimator_thresholds[float('inf')],
                    self.take_profit_multipliers[float('inf')],
                    self.stop_loss_multipliers[float('inf')],
                    position_pct
                )

            def compute_optimal_pattern(self, start_idx, end_idx):
                prices = self.data['close'].iloc[start_idx:end_idx].values
                if len(prices) < 3:
                    print(
                        f"Warning: Not enough data for pattern ({len(prices)} points) in window {start_idx}-{end_idx}")
                    return pd.Series(np.zeros(max(0, len(prices)-1)), index=self.data.index[start_idx:end_idx-1])

                returns = pd.Series(prices).pct_change().shift().dropna()
                if len(returns) == 0:
                    print(
                        f"Warning: No valid returns in window {start_idx}-{end_idx}")
                    return pd.Series(np.zeros(len(prices)-1), index=self.data.index[start_idx:end_idx-1])

                threshold, _, _, _, _ = self.get_volatility_params(prices)
                pattern = np.where(returns > threshold, 1,
                                   np.where(returns < -threshold, -1, 0))
                return pd.Series(pattern, index=self.data.index[start_idx:start_idx + len(pattern)])

            def prepare_features(self, start_idx, end_idx):
                if start_idx == 0:
                    window_data = self.data.iloc[:end_idx]
                else:
                    window_data = self.data.iloc[:end_idx-1]
                window_index = self.data.index[start_idx:end_idx]
                prices_btc = window_data['close'].values

                if len(prices_btc) < 2:
                    print(
                        f"Warning: Not enough data for features ({len(prices_btc)} points) in window {start_idx}-{end_idx}")
                    return np.zeros((len(window_index), 22))

                hurst_btc = []
                fdi_btc = []
                entropy_btc = []
                min_window = min(20, len(prices_btc))
                for i in range(max(0, len(prices_btc) - len(window_index)), len(prices_btc)):
                    lookback = prices_btc[max(0, i - min_window + 1):i + 1]
                    hurst_btc.append(
                        self.btc_indicators.hurst_exponent(lookback))
                    fdi_btc.append(self.btc_indicators.fdi(lookback))
                    entropy_btc.append(self.btc_indicators.entropy(lookback))

                hurst_btc = pd.Series(hurst_btc, index=window_index)
                fdi_btc = pd.Series(fdi_btc, index=window_index)
                entropy_btc = pd.Series(entropy_btc, index=window_index)

                upper_btc, middle_btc, lower_btc = self.btc_indicators.bollinger_bands(
                    prices_btc)
                bb_width_btc = (upper_btc[-len(window_index):] - lower_btc[-len(
                    window_index):]) / middle_btc[-len(window_index):]
                macd_btc = self.btc_indicators.macd(
                    prices_btc)[-len(window_index):]
                sma_btc = self.btc_indicators.sma(
                    prices_btc)[-len(window_index):]
                ema_btc = self.btc_indicators.ema(
                    prices_btc)[-len(window_index):]
                rsi_btc = self.btc_indicators.rsi(
                    prices_btc)[-len(window_index):]
                obv_btc = self.btc_indicators.obv(
                    prices_btc, window_data['volume'].values)[-len(window_index):]
                slowk_btc = self.btc_indicators.stochastic(
                    window_data.get('high_btc', window_data['close']).values,
                    window_data.get('low_btc', window_data['close']).values,
                    prices_btc
                )[-len(window_index):]
                vwap_btc = self.btc_indicators.vwap(
                    window_data.get('high_btc', window_data['close']).values,
                    window_data.get('low_btc', window_data['close']).values,
                    prices_btc,
                    window_data['volume'].values
                )[-len(window_index):]

                v_hurst_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(hurst_btc.values)]), index=window_index)
                v_fdi_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(fdi_btc.values)]), index=window_index)
                v_bb_width_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(bb_width_btc.values)]), index=window_index)
                v_entropy_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(entropy_btc.values)]), index=window_index)
                v_macd_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(macd_btc.values)]), index=window_index)
                v_sma_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(sma_btc.values)]), index=window_index)
                v_ema_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(ema_btc.values)]), index=window_index)
                v_rsi_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(rsi_btc.values)]), index=window_index)
                v_obv_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(obv_btc.values)]), index=window_index)
                v_slowk_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(slowk_btc.values)]), index=window_index)
                v_vwap_btc = pd.Series(np.concatenate(
                    [[0], self.btc_indicators.velocity(vwap_btc.values)]), index=window_index)

                features = pd.DataFrame(index=window_index)
                features['hurst_btc'] = hurst_btc
                features['fdi_btc'] = fdi_btc
                features['bb_width_btc'] = bb_width_btc
                features['entropy_btc'] = entropy_btc
                features['macd_btc'] = macd_btc
                features['sma_btc'] = sma_btc
                features['ema_btc'] = ema_btc
                features['rsi_btc'] = rsi_btc
                features['obv_btc'] = obv_btc
                features['slowk_btc'] = slowk_btc
                features['vwap_btc'] = vwap_btc
                features['v_hurst_btc'] = v_hurst_btc
                features['v_fdi_btc'] = v_fdi_btc
                features['v_bb_width_btc'] = v_bb_width_btc
                features['v_entropy_btc'] = v_entropy_btc
                features['v_macd_btc'] = v_macd_btc
                features['v_sma_btc'] = v_sma_btc
                features['v_ema_btc'] = v_ema_btc
                features['v_rsi_btc'] = v_rsi_btc
                features['v_obv_btc'] = v_obv_btc
                features['v_slowk_btc'] = v_slowk_btc
                features['v_vwap_btc'] = v_vwap_btc

                return self.scaler.fit_transform(features.replace([np.inf, -np.inf], np.nan).fillna(0))

            def train_random_forest(self, start_idx, end_idx):
                features = self.prepare_features(start_idx, end_idx)
                pattern = self.compute_optimal_pattern(start_idx, end_idx)
                X = features[:len(pattern)]
                y = pattern.values

                if len(X) < 5:
                    print(
                        f"Insufficient data ({len(X)} samples) to train BTC RF in window {start_idx}-{end_idx}")
                    return False
                if len(np.unique(y)) < 2:
                    print(
                        f"Warning: No variability in target (all {np.unique(y)[0]}) in window {start_idx}-{end_idx}")

                prices = self.data['close'].iloc[start_idx:end_idx].values
                _, n_estimators, _, _, _ = self.get_volatility_params(prices)
                self.btc_rf = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=6, random_state=19)
                self.btc_rf.fit(X, y)
                self.is_rf_fitted = True
                return True

            def mask_signals(self, raw_signals, pred_index):
                masked_signals = np.zeros_like(raw_signals)
                signals_df = pd.DataFrame(index=pred_index)
                prices = self.data['close'].loc[pred_index].values
                atr_values = self.atr.loc[pred_index].values

                start_idx = self.data.index.get_loc(pred_index[0])
                end_idx = self.data.index.get_loc(pred_index[-1]) + 1
                window_prices = self.data['close'].iloc[max(
                    0, start_idx - self.t_recalc):end_idx].values
                if len(window_prices) < 2:
                    take_profit_mult = self.base_take_profit_multiplier
                    stop_loss_mult = self.base_stop_loss_multiplier
                else:
                    _, _, take_profit_mult, stop_loss_mult, _ = self.get_volatility_params(
                        window_prices)

                for i in range(len(raw_signals)):
                    current_price = prices[i]
                    current_atr = atr_values[i]
                    take_profit = current_atr * take_profit_mult
                    stop_loss = current_atr * stop_loss_mult

                    if self.stop_loss_cooldown > 0:
                        self.stop_loss_cooldown -= 1
                        masked_signals[i] = 0
                        continue

                    if self.position == 1:
                        if current_price >= self.entry_price + take_profit:
                            masked_signals[i] = -1
                            self.position = 0
                            self.entry_price = None
                        elif current_price <= self.entry_price - stop_loss:
                            masked_signals[i] = -1
                            self.position = 0
                            self.entry_price = None
                            self.stop_loss_cooldown = self.max_cooldown
                    elif self.position == -1:
                        if current_price <= self.entry_price - take_profit:
                            masked_signals[i] = 1
                            self.position = 0
                            self.entry_price = None
                        elif current_price >= self.entry_price + stop_loss:
                            masked_signals[i] = 1
                            self.position = 0
                            self.entry_price = None
                            self.stop_loss_cooldown = self.max_cooldown

                    if self.position == 0 and masked_signals[i] == 0:
                        if raw_signals[i] == 1:
                            masked_signals[i] = 1
                            self.position = 1
                            self.entry_price = current_price
                        elif raw_signals[i] == -1:
                            masked_signals[i] = -1
                            self.position = -1
                            self.entry_price = current_price
                    elif self.position == 1 and raw_signals[i] == -1 and masked_signals[i] == 0:
                        masked_signals[i] = -1
                        self.position = 0
                        self.entry_price = None
                    elif self.position == -1 and raw_signals[i] == 1 and masked_signals[i] == 0:
                        masked_signals[i] = 1
                        self.position = 0
                        self.entry_price = None

                signals_df['buy_btc'] = masked_signals == 1
                signals_df['sell_btc'] = masked_signals == -1
                return signals_df

            def compute_signals(self):
                self.signals['buy_btc'] = False
                self.signals['sell_btc'] = False
                self.position = 0
                self.stop_loss_cooldown = 0

                initial_end_idx = min(self.t_recalc, len(self.data))
                if initial_end_idx > 1:
                    success = self.train_random_forest(0, initial_end_idx)
                    if not success:
                        self.signals.loc[self.data.index[:initial_end_idx],
                                         'buy_btc'] = False
                        self.signals.loc[self.data.index[:initial_end_idx],
                                         'sell_btc'] = False
                    else:
                        pred_end_idx = min(
                            initial_end_idx + self.t_recalc, len(self.data))
                        pred_features = self.prepare_features(
                            initial_end_idx, pred_end_idx)
                        pred_index = self.data.index[initial_end_idx:initial_end_idx + len(
                            pred_features)]
                        btc_signals = self.btc_rf.predict(pred_features)
                        masked_signals_df = self.mask_signals(
                            btc_signals, pred_index)
                        self.signals.loc[pred_index,
                                         'buy_btc'] = masked_signals_df['buy_btc']
                        self.signals.loc[pred_index,
                                         'sell_btc'] = masked_signals_df['sell_btc']

                for i in range(self.t_recalc, len(self.data), self.t_recalc):
                    start_idx = i - self.t_recalc
                    end_idx = i
                    pred_start_idx = i
                    pred_end_idx = min(i + self.t_recalc, len(self.data))

                    if end_idx - start_idx >= 5:
                        success = self.train_random_forest(start_idx, end_idx)
                        if not success and not self.is_rf_fitted:
                            self.signals.loc[self.data.index[pred_start_idx:pred_end_idx],
                                             'buy_btc'] = False
                            self.signals.loc[self.data.index[pred_start_idx:pred_end_idx],
                                             'sell_btc'] = False
                            continue

                    if pred_end_idx > pred_start_idx and self.is_rf_fitted:
                        pred_features = self.prepare_features(
                            pred_start_idx, pred_end_idx)
                        pred_index = self.data.index[pred_start_idx:pred_start_idx + len(
                            pred_features)]
                        btc_signals = self.btc_rf.predict(pred_features)
                        masked_signals_df = self.mask_signals(
                            btc_signals, pred_index)
                        self.signals.loc[pred_index,
                                         'buy_btc'] = masked_signals_df['buy_btc']
                        self.signals.loc[pred_index,
                                         'sell_btc'] = masked_signals_df['sell_btc']

                last_start_idx = (len(self.data) //
                                  self.t_recalc) * self.t_recalc
                if last_start_idx < len(self.data) and self.is_rf_fitted:
                    pred_features = self.prepare_features(
                        last_start_idx, len(self.data))
                    pred_index = self.data.index[last_start_idx:last_start_idx + len(
                        pred_features)]
                    btc_signals = self.btc_rf.predict(pred_features)
                    masked_signals_df = self.mask_signals(
                        btc_signals, pred_index)
                    self.signals.loc[pred_index,
                                     'buy_btc'] = masked_signals_df['buy_btc']
                    self.signals.loc[pred_index,
                                     'sell_btc'] = masked_signals_df['sell_btc']

        class Backtest:
            def __init__(self, strategy, initial_capital=10000, fee=0.001):
                self.strategy = strategy
                self.data = strategy.data
                self.signals = strategy.signals
                self.initial_capital = initial_capital
                self.fee = fee
                self.capital = initial_capital
                self.btc_position = 0
                self.portfolio_value = []
                self.output_data = pd.DataFrame(index=self.data.index)

            def compute_metrics(self):
                returns = pd.Series(self.portfolio_value).pct_change().dropna()
                total_return = (
                    self.portfolio_value[-1] - self.initial_capital) / self.initial_capital
                sharpe_ratio = np.mean(
                    returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                drawdowns = pd.Series(
                    self.portfolio_value).cummax() - self.portfolio_value
                max_drawdown = drawdowns.max() / pd.Series(self.portfolio_value).cummax().max()
                trades = self.output_data['signal'].diff().abs().sum() / 2
                wins = len(self.output_data[(self.output_data['signal'] == 1) & (self.output_data['close'].shift(-1) > self.output_data['close'])] +
                           self.output_data[(self.output_data['signal'] == -1) & (self.output_data['close'].shift(-1) < self.output_data['close'])])
                win_rate = wins / trades if trades > 0 else 0
                return {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'trades': trades
                }

            def run(self):
                self.output_data['index'] = range(len(self.data))
                self.output_data['datetime'] = self.data['datetime']
                self.output_data['open'] = self.data.get(
                    'open_btc', self.data['close'])
                self.output_data['high'] = self.data.get(
                    'high_btc', self.data['close'])
                self.output_data['low'] = self.data.get(
                    'low_btc', self.data['close'])
                self.output_data['close'] = self.data['close']
                self.output_data['signal'] = 0
                self.output_data['trade_type'] = 'HOLD'
                self.output_data['volume'] = self.data.get('volume', 0)
                # Default position percentage
                self.output_data['position'] = 60.0

                for i in range(1, len(self.data)):
                    btc_price = self.data['close'].iloc[i]
                    # Calculate position % based on volatility over a rolling window
                    window_prices = self.data['close'].iloc[max(
                        0, i - self.strategy.t_recalc):i + 1].values
                    _, _, _, _, position_pct = self.strategy.get_volatility_params(
                        window_prices)
                    self.output_data.loc[self.data.index[i],
                                         'position'] = position_pct

                    if self.signals['buy_btc'].iloc[i] and self.btc_position == 0 and self.capital > 0:
                        btc_buy = self.capital * (1 - self.fee) / btc_price
                        self.btc_position = btc_buy
                        self.capital -= btc_buy * btc_price
                        self.output_data.loc[self.data.index[i], 'signal'] = 1
                        self.output_data.loc[self.data.index[i],
                                             'trade_type'] = 'LONG'
                    elif self.signals['sell_btc'].iloc[i] and self.btc_position == 0 and self.capital > 0:
                        btc_sell = self.capital * (1 - self.fee) / btc_price
                        self.btc_position = -btc_sell
                        self.capital += btc_sell * btc_price
                        self.output_data.loc[self.data.index[i], 'signal'] = -1
                        self.output_data.loc[self.data.index[i],
                                             'trade_type'] = 'SHORT'
                    elif self.signals['sell_btc'].iloc[i] and self.btc_position > 0:
                        self.capital += self.btc_position * \
                            btc_price * (1 - self.fee)
                        self.btc_position = 0
                        self.output_data.loc[self.data.index[i], 'signal'] = -1
                        self.output_data.loc[self.data.index[i],
                                             'trade_type'] = 'CLOSE'
                    elif self.signals['buy_btc'].iloc[i] and self.btc_position < 0:
                        cost = abs(self.btc_position) * \
                            btc_price * (1 + self.fee)
                        self.capital -= cost
                        self.btc_position = 0
                        self.output_data.loc[self.data.index[i], 'signal'] = 1
                        self.output_data.loc[self.data.index[i],
                                             'trade_type'] = 'CLOSE'

                    portfolio_val = self.capital + \
                        (self.btc_position * btc_price)
                    self.portfolio_value.append(portfolio_val)

                self.portfolio_value = [
                    self.initial_capital] + self.portfolio_value
                metrics = self.compute_metrics()
                print(f"Backtest completed: {metrics}")
                return self.output_data

        strategy = TradingStrategy(data, vol_thresholds, estimator_thresholds,
                                   base_take_profit_multiplier, base_stop_loss_multiplier)
        strategy.compute_signals()
        backtest = Backtest(strategy)
        signals = backtest.run()
        return signals

    def tune_thresholds(self, data: pd.DataFrame, param_grid: dict, initial_capital=10000, fee=0.001):
        results = []
        for vol_thresh, est_thresh, tp_mult, sl_mult in product(
            param_grid['vol_thresholds'],
            param_grid['estimator_thresholds'],
            param_grid['take_profit_multiplier'],
            param_grid['stop_loss_multiplier']
        ):
            signals = self.run(data, vol_thresh, est_thresh, tp_mult, sl_mult)
            backtest = Backtest(TradingStrategy(
                data, vol_thresh, est_thresh, tp_mult, sl_mult), initial_capital, fee)
            backtest.run()
            metrics = backtest.compute_metrics()
            results.append({
                'vol_thresholds': vol_thresh,
                'estimator_thresholds': est_thresh,
                'take_profit_multiplier': tp_mult,
                'stop_loss_multiplier': sl_mult,
                **metrics
            })
            print(f"Tested: TP={tp_mult}, SL={sl_mult}, Metrics={metrics}")

        results_df = pd.DataFrame(results)
        return results_df

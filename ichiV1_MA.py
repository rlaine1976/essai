# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair
import numpy as np
from freqtrade.strategy import stoploss_from_open
from freqtrade.strategy import IntParameter, DecimalParameter, CategoricalParameter


class ichiV1_MA(IStrategy):
    """
    Variante de ichiV1 : ajoute un signal supplementaire base sur le croisement
    de deux moyennes mobiles (EMA rapide / EMA lente), combine en OU avec les
    conditions Ichimoku existantes. Un trade peut donc se declencher soit sur
    le setup Ichimoku d'origine, soit sur un croisement de moyennes mobiles,
    sans que l'un empeche l'autre.

    Nom de classe/fichier different de l'original pour ne pas ecraser
    user_data/strategies/ichiV1.json (parametres hyperopt de la version de base).
    """

    # NOTE: valeurs par défaut assouplies pour augmenter la fréquence de trading.
    # Avec les réglages d'origine (EMA50/EMA200 + conditions Ichimoku strictes),
    # le croisement de moyennes mobiles ne se produit quasiment jamais sur 5m
    # (observé : 1 seul trade en 4 jours de dry-run) : EMA50/EMA200 se croisent
    # peut-être une fois toutes les quelques semaines, et le setup Ichimoku
    # complet (niveau 2 + niveau 4 + magnitude) reste rare aussi.
    buy_params = {
        "buy_fan_magnitude_shift_value": 1,
        "buy_min_fan_magnitude_gain": 1.000,
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 1,
        "buy_ma_fast_period": 20,
        "buy_ma_slow_period": 60,
    }
    sell_params = {
        "sell_trend_indicator": "trend_close_30m",
    }

    # --- Hyperoptable buy-space parameters (Ichimoku, inchanges) ---
    buy_trend_above_senkou_level = IntParameter(
        1, 8, default=buy_params["buy_trend_above_senkou_level"], space="buy", optimize=True, load=True
    )
    buy_trend_bullish_level = IntParameter(
        1, 8, default=buy_params["buy_trend_bullish_level"], space="buy", optimize=True, load=True
    )
    buy_fan_magnitude_shift_value = IntParameter(
        1, 10, default=buy_params["buy_fan_magnitude_shift_value"], space="buy", optimize=True, load=True
    )
    buy_min_fan_magnitude_gain = DecimalParameter(
        1.000, 1.020, default=buy_params["buy_min_fan_magnitude_gain"], decimals=3,
        space="buy", optimize=True, load=True
    )

    # --- Nouveaux parametres hyperoptables pour les moyennes mobiles ---
    # Periode de l'EMA rapide (nombre de bougies 5m). Plage resserree (5-20 au
    # lieu de 10-100) pour garantir des croisements frequents plutot que
    # quelques-uns par mois.
    buy_ma_fast_period = IntParameter(
        5, 20, default=buy_params["buy_ma_fast_period"], space="buy", optimize=True, load=True
    )
    # Periode de l'EMA lente (nombre de bougies 5m). Plage resserree (15-60
    # au lieu de 100-400) pour la meme raison.
    buy_ma_slow_period = IntParameter(
        15, 60, default=buy_params["buy_ma_slow_period"], space="buy", optimize=True, load=True
    )

    # --- Hyperoptable sell-space parameter ---
    sell_trend_indicator = CategoricalParameter(
        [
            "trend_close_5m", "trend_close_15m", "trend_close_30m", "trend_close_1h",
            "trend_close_2h", "trend_close_4h", "trend_close_6h", "trend_close_8h",
        ],
        default=sell_params["sell_trend_indicator"], space="sell", optimize=True, load=True
    )

    # ROI table:
    minimal_roi = {
        "0": 0.036,
        "32": 0.026,
        "91": 0.014,
        "193": 0
    }

    # Stoploss:
    stoploss = -0.229

    # Optimal timeframe for the strategy
    timeframe = '5m'
    # Ichimoku (base_line_periods=60 + displacement=30) est maintenant le facteur
    # limitant, plus large que buy_ma_slow_period (max 60 desormais) : 120 de marge.
    startup_candle_count = 120
    process_only_new_candles = False

    # Trailing stop parameters:
    trailing_stop = True
    trailing_stop_positive = 0.045
    trailing_stop_positive_offset = 0.107
    trailing_only_offset_is_reached = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    plot_config = {
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(255,76,46,0.2)',
            },
            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'},
            'ma_fast': {'color': 'blue'},
            'ma_slow': {'color': 'orange'},
        },
        'subplots': {
            'fan_magnitude': {
                'fan_magnitude': {}
            },
            'fan_magnitude_gain': {
                'fan_magnitude_gain': {}
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['close'], timeperiod=72)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['close'], timeperiod=96)

        dataframe['trend_open_5m'] = dataframe['open']
        dataframe['trend_open_15m'] = ta.EMA(dataframe['open'], timeperiod=3)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['open'], timeperiod=6)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['open'], timeperiod=12)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['open'], timeperiod=24)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['open'], timeperiod=48)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['open'], timeperiod=72)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['open'], timeperiod=96)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['chikou_span'] = ichimoku['chikou_span']
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green']
        dataframe['cloud_red'] = ichimoku['cloud_red']

        dataframe['atr'] = ta.ATR(dataframe)

        # --- Moyennes mobiles pour le signal additionnel ---
        # Note: recalculees a chaque essai de hyperopt puisque les periodes
        # sont hyperoptables (buy_ma_fast_period / buy_ma_slow_period).
        dataframe['ma_fast'] = ta.EMA(dataframe['close'], timeperiod=self.buy_ma_fast_period.value)
        dataframe['ma_slow'] = ta.EMA(dataframe['close'], timeperiod=self.buy_ma_slow_period.value)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Trending market
        if self.buy_trend_above_senkou_level.value >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])
        if self.buy_trend_above_senkou_level.value >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_b'])
        if self.buy_trend_above_senkou_level.value >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_b'])
        if self.buy_trend_above_senkou_level.value >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_b'])
        if self.buy_trend_above_senkou_level.value >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_b'])
        if self.buy_trend_above_senkou_level.value >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_b'])
        if self.buy_trend_above_senkou_level.value >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_b'])
        if self.buy_trend_above_senkou_level.value >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_b'])

        # Trends bullish
        if self.buy_trend_bullish_level.value >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])
        if self.buy_trend_bullish_level.value >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['trend_open_15m'])
        if self.buy_trend_bullish_level.value >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['trend_open_30m'])
        if self.buy_trend_bullish_level.value >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['trend_open_1h'])
        if self.buy_trend_bullish_level.value >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['trend_open_2h'])
        if self.buy_trend_bullish_level.value >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['trend_open_4h'])
        if self.buy_trend_bullish_level.value >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['trend_open_6h'])
        if self.buy_trend_bullish_level.value >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['trend_open_8h'])

        # Trends magnitude
        conditions.append(dataframe['fan_magnitude_gain'] >= self.buy_min_fan_magnitude_gain.value)
        conditions.append(dataframe['fan_magnitude'] > 1)
        for x in range(self.buy_fan_magnitude_shift_value.value):
            conditions.append(dataframe['fan_magnitude'].shift(x + 1) < dataframe['fan_magnitude'])

        # --- Signal Ichimoku d'origine (toutes conditions ci-dessus, en ET) ---
        ichimoku_signal = reduce(lambda x, y: x & y, conditions) if conditions else False

        # --- Nouveau signal additionnel : croisement haussier EMA rapide / EMA lente ---
        ma_cross_signal = qtpylib.crossed_above(dataframe['ma_fast'], dataframe['ma_slow'])

        dataframe.loc[
            (ichimoku_signal | ma_cross_signal) & (dataframe['volume'] > 0),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_trend_indicator.value]))

        ichimoku_exit = reduce(lambda x, y: x & y, conditions) if conditions else False

        # --- Signal de sortie additionnel : croisement baissier EMA rapide / EMA lente ---
        ma_cross_down = qtpylib.crossed_below(dataframe['ma_fast'], dataframe['ma_slow'])

        dataframe.loc[
            (ichimoku_exit | ma_cross_down) & (dataframe['volume'] > 0),
            'sell'] = 1

        return dataframe

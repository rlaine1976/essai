# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
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


class ichiV1(IStrategy):
    # NOTE: settings as of the 25th july 21

    # MIGRATION API (27 juillet 2026) : passage à l'API moderne Freqtrade
    # (populate_buy_trend/populate_sell_trend + colonnes 'buy'/'sell' ->
    # populate_entry_trend/populate_exit_trend + colonnes 'enter_long'/
    # 'exit_long', INTERFACE_VERSION = 3, use_sell_signal -> use_exit_signal,
    # sell_profit_only -> exit_profit_only, ignore_roi_if_buy_signal ->
    # ignore_roi_if_entry_signal). L'ancienne API restait supportée en
    # compatibilité mais est dépréciée ; ce changement n'affecte pas la
    # logique de trading, seulement l'interface.

    # ASSOUPLISSEMENT (27 juillet 2026) — voir explication détaillée ci-dessous
    # -------------------------------------------------------------------------
    # Constat : 0 trade en ~10 jours de dry-run alors que le backtest sur 1 an
    # donnait ~0.5 trade/jour. En inspectant la logique, le signal d'achat
    # empile ~21 conditions simultanées :
    #   - alignement haussier au-dessus du nuage Ichimoku sur 6 horizons
    #     (5m/15m/30m/1h/2h/4h) = 12 conditions (buy_trend_above_senkou_level=6)
    #   - tendance haussière (close > open) sur les 6 mêmes horizons
    #     = 6 conditions (buy_trend_bullish_level=6)
    #   - accélération du ratio EMA 1h/8h sur 3 bougies consécutives
    #     (buy_fan_magnitude_shift_value=3)
    # C'est cohérent avec la moyenne de 0.5 trade/jour sur un an (stratégie
    # volontairement sélective), mais sur un marché récent hésitant/baissier
    # (BTC en zone de peur, MM50 au-dessus du prix et baissière mi-juillet
    # 2026), un alignement simultané sur 6 horizons ne s'est simplement pas
    # produit. Ce n'était donc probablement pas un bug.
    #
    # Pour rendre la stratégie plus souple (plus de trades, quitte à baisser
    # un peu la sélectivité), les 3 paramètres suivants ont été réduits :
    #   - buy_trend_above_senkou_level : 6 -> 3 (n'exige l'alignement
    #     au-dessus du nuage que sur 5m/15m/30m, plus jusqu'à 4h)
    #   - buy_trend_bullish_level : 6 -> 3 (idem pour la tendance haussière)
    #   - buy_fan_magnitude_shift_value : 3 -> 1 (n'exige plus que 1 bougie
    #     d'accélération du momentum au lieu de 3 consécutives)
    # buy_min_fan_magnitude_gain est conservé à 1.002 (déjà la valeur "plus de
    # trades, ~70% win rate" selon le commentaire d'origine) pour ne pas trop
    # dégrader la qualité des signaux en une seule fois.
    #
    # À REBACKTESTER avant de relancer en dry-run, pour comparer le nombre de
    # trades et le win rate à la version d'origine (level=6/6, shift=3).
    # -------------------------------------------------------------------------
    #
    # RÉSULTAT du backtest avec ces réglages assouplis (2024-01-01 -> 2026-07-
    # 18, BTC/ETH/SOL/XRP-USDC) : +16.20% de profit total (marché à -21.95%
    # sur la même période), 0.87 trade/jour, Sharpe 2.96, drawdown max 1.93%.
    # Détail par raison de sortie : les 452 sorties ROI sont excellentes
    # (92.9% gagnantes, +41.53% de contribution), mais les 355 sorties par
    # exit_signal (croisement trend_close_5m sous trend_close_2h) sont
    # mauvaises (5.4% gagnantes, -25.33% de contribution) — le signal de
    # vente mange une bonne partie du gain généré par le ROI.
    #
    # AMÉLIORATION DU SIGNAL DE SORTIE (27 juillet 2026) :
    # exit_profit_only passé à True. Le signal de vente (croisement baissier)
    # ne pourra désormais se déclencher QUE si le trade est déjà profitable —
    # il servira à sécuriser un gain avant qu'il ne s'évapore, au lieu de
    # pouvoir couper un trade en perte comme c'était le cas (94.6% des
    # sorties par signal étaient perdantes).
    #
    # RÉSULTAT : ce changement a dégradé la performance (+9.26% au lieu de
    # +16.20%, Sharpe 0.66 au lieu de 2.96). Explication : le signal de vente,
    # malgré son faible taux de réussite, servait en réalité de coupe-perte
    # précoce (355 sorties à -1.11% en moyenne, -1139.87 USDC au total). En
    # l'empêchant de couper les trades perdants, ceux-ci restent ouverts
    # jusqu'au stoploss (-27.5%, très large) : seulement 13 trades ont fini
    # au stoploss, mais à -28.08% de moyenne chacun (-1067.34 USDC au total)
    # — une perte totale similaire, concentrée sur beaucoup moins de trades
    # avec un risque par trade bien plus violent (pire trade : -28.08%).
    #
    # AJUSTEMENT (27 juillet 2026) : stoploss resserré de -27.5% à -8% pour
    # limiter la casse maintenant que le signal ne le fait plus sur les
    # trades perdants. Résultat : mieux que la combinaison précédente
    # (+14.56%, Sharpe 1.75, drawdown 2.40%), mais toujours en dessous de la
    # configuration d'origine (+16.20%, Sharpe 2.96, drawdown 1.93%).
    #
    # CONCLUSION (27 juillet 2026) : sur 3 configurations testées, l'origine
    # (exit_profit_only=False, stoploss=-27.5%) reste la meilleure. Le signal
    # de vente, malgré un faible taux de réussite affiché isolément (5.4%),
    # apportait une vraie valeur en tant que coupe-perte rapide sur les
    # trades perdants — un stoploss large derrière lui ne servait
    # quasiment jamais. Revert : exit_profit_only -> False, stoploss ->
    # -0.275 (valeurs d'origine).

    INTERFACE_VERSION = 3

    # Buy hyperspace params:
    buy_params = {
        "buy_trend_above_senkou_level": 3,   # était 6
        "buy_trend_bullish_level": 3,        # était 6
        "buy_fan_magnitude_shift_value": 1,  # était 3
        "buy_min_fan_magnitude_gain": 1.002  # NOTE: Good value (Win% ~70%), alot of trades
        #"buy_min_fan_magnitude_gain": 1.008 # NOTE: Very save value (Win% ~90%), only the biggest moves 1.008,
    }
    # Sell hyperspace params:
    # NOTE: was 15m but kept bailing out in dryrun
    sell_params = {
        "sell_trend_indicator": "trend_close_2h",
    }
    # ROI table:
    minimal_roi = {
        "0": 0.059,
        "10": 0.037,
        "41": 0.012,
        "114": 0
    }
    # Stoploss:
    stoploss = -0.275
    # Optimal timeframe for the strategy
    timeframe = '5m'
    startup_candle_count = 96
    process_only_new_candles = False
    trailing_stop = False
    #trailing_stop_positive = 0.002
    #trailing_stop_positive_offset = 0.025
    #trailing_only_offset_is_reached = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    plot_config = {
        'main_plot': {
            # fill area between senkou_a and senkou_b
            'senkou_a': {
                'color': 'green', #optional
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud', #optional
                'fill_color': 'rgba(255,76,46,0.2)', #optional
            },
            # plot senkou_b, too. Not only the area to it.
            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'}
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
        #dataframe['close'] = heikinashi['close']
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

        # SUIVI TELEGRAM (27 juillet 2026) : notifie le nombre de conditions
        # d'entrée actuellement validées sur les 6 nécessaires, pour suivre
        # la progression vers un signal d'achat sans attendre qu'il se
        # déclenche. Envoie un message uniquement quand le compte change
        # (pas à chaque itération) pour éviter le spam. self.dp.send_msg()
        # ne fait rien en backtest/hyperopt, seulement en live/dry-run.
        self._notify_entry_conditions(dataframe, metadata['pair'])

        return dataframe

    def _notify_entry_conditions(self, dataframe: DataFrame, pair: str) -> None:
        if dataframe.empty:
            return
        last = dataframe.iloc[-1]

        timeframes = ['5m', '15m', '30m', '1h', '2h', '4h']
        senkou_ok = sum(
            1 for tf in timeframes
            if last[f'trend_close_{tf}'] > last['senkou_a'] and last[f'trend_close_{tf}'] > last['senkou_b']
        )
        bullish_ok = sum(
            1 for tf in timeframes
            if last[f'trend_close_{tf}'] > last[f'trend_open_{tf}']
        )
        fan_conditions = [
            last['fan_magnitude_gain'] >= self.buy_params['buy_min_fan_magnitude_gain'],
            last['fan_magnitude'] > 1,
        ]
        for x in range(self.buy_params['buy_fan_magnitude_shift_value']):
            fan_conditions.append(dataframe['fan_magnitude'].shift(x + 1).iloc[-1] < last['fan_magnitude'])
        fan_ok = all(bool(c) for c in fan_conditions)

        if not hasattr(self, '_last_condition_counts'):
            self._last_condition_counts = {}

        current = (senkou_ok, bullish_ok, fan_ok)
        if self._last_condition_counts.get(pair) == current:
            return
        self._last_condition_counts[pair] = current

        senkou_needed = self.buy_params['buy_trend_above_senkou_level']
        bullish_needed = self.buy_params['buy_trend_bullish_level']
        self.dp.send_msg(
            f"[{pair}] ichiV1 — conditions d'entrée : "
            f"nuage {senkou_ok}/6 (seuil {senkou_needed}) · "
            f"tendance haussière {bullish_ok}/6 (seuil {bullish_needed}) · "
            f"momentum {'OK' if fan_ok else 'non'}"
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # Trending market
        if self.buy_params['buy_trend_above_senkou_level'] >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])
        if self.buy_params['buy_trend_above_senkou_level'] >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_b'])
        if self.buy_params['buy_trend_above_senkou_level'] >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_b'])
        if self.buy_params['buy_trend_above_senkou_level'] >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_b'])
        if self.buy_params['buy_trend_above_senkou_level'] >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_b'])
        if self.buy_params['buy_trend_above_senkou_level'] >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_b'])
        if self.buy_params['buy_trend_above_senkou_level'] >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_b'])
        if self.buy_params['buy_trend_above_senkou_level'] >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_b'])
        # Trends bullish
        if self.buy_params['buy_trend_bullish_level'] >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])
        if self.buy_params['buy_trend_bullish_level'] >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['trend_open_15m'])
        if self.buy_params['buy_trend_bullish_level'] >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['trend_open_30m'])
        if self.buy_params['buy_trend_bullish_level'] >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['trend_open_1h'])
        if self.buy_params['buy_trend_bullish_level'] >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['trend_open_2h'])
        if self.buy_params['buy_trend_bullish_level'] >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['trend_open_4h'])
        if self.buy_params['buy_trend_bullish_level'] >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['trend_open_6h'])
        if self.buy_params['buy_trend_bullish_level'] >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['trend_open_8h'])
        # Trends magnitude
        conditions.append(dataframe['fan_magnitude_gain'] >= self.buy_params['buy_min_fan_magnitude_gain'])
        conditions.append(dataframe['fan_magnitude'] > 1)
        for x in range(self.buy_params['buy_fan_magnitude_shift_value']):
            conditions.append(dataframe['fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_params['sell_trend_indicator']]))
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        return dataframe

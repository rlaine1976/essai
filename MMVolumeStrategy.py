"""
MMVolumeStrategy — stratégie Freqtrade

LOGIQUE (telle que demandée) :

ACHAT (entrée long) :
  - La moyenne mobile (SMA) est ascendante depuis 2 bougies
    (sma[t] > sma[t-1] > sma[t-2])
  - ET le volume est ascendant (volume[t] > volume[t-1])

VENTE (sortie long) :
  - La moyenne mobile "coupe" une bougie baissière : le prix de clôture passe
    d'au-dessus à en-dessous de la SMA (croisement baissier close/SMA) sur une
    bougie rouge (close < open)
  - ET il y a 2 bougies baissières d'affilée (la bougie du croisement + la précédente)

Hypothèses faites faute de précision (facilement modifiables ci-dessous) :
  - "moyenne mobile" = SMA simple, période 20 (MA_PERIOD)
  - "volume ascendant" = volume de la bougie en cours > volume de la bougie précédente
    (les données OHLCV de Freqtrade n'ont qu'une seule colonne "volume", pas de
    volume acheteur/vendeur séparé — c'est donc ce volume qui est utilisé)
  - timeframe : 5m

À tester en backtest avant tout usage en réel :
  freqtrade backtesting --strategy MMVolumeStrategy --timeframe 5m
"""

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class MMVolumeStrategy(IStrategy):

    INTERFACE_VERSION = 3

    # Période de la moyenne mobile
    MA_PERIOD = 20

    # ROI / stoploss / timeframe — à ajuster selon votre profil de risque
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0
    }
    stoploss = -0.10
    trailing_stop = False

    timeframe = "5m"

    startup_candle_count: int = MA_PERIOD + 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Moyenne mobile simple
        dataframe["sma"] = ta.SMA(dataframe, timeperiod=self.MA_PERIOD)

        # Bougie baissière = clôture sous l'ouverture
        dataframe["bearish_candle"] = dataframe["close"] < dataframe["open"]

        # SMA ascendante depuis 2 bougies : sma[t] > sma[t-1] > sma[t-2]
        dataframe["sma_rising_2"] = (
            (dataframe["sma"] > dataframe["sma"].shift(1))
            & (dataframe["sma"].shift(1) > dataframe["sma"].shift(2))
        )

        # Volume ascendant (bougie en cours > bougie précédente)
        dataframe["volume_rising"] = dataframe["volume"] > dataframe["volume"].shift(1)

        # Croisement baissier : close était au-dessus de la SMA, passe en-dessous
        dataframe["cross_down_sma"] = (
            (dataframe["close"].shift(1) > dataframe["sma"].shift(1))
            & (dataframe["close"] < dataframe["sma"])
        )

        # 2 bougies baissières d'affilée
        dataframe["two_bearish_in_a_row"] = (
            dataframe["bearish_candle"] & dataframe["bearish_candle"].shift(1)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["sma_rising_2"]
                & dataframe["volume_rising"]
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["cross_down_sma"]
                & dataframe["bearish_candle"]
                & dataframe["two_bearish_in_a_row"]
            ),
            "exit_long",
        ] = 1

        return dataframe

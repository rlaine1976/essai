"""
MMVolumeStrategy — stratégie Freqtrade

LOGIQUE (telle que demandée) :

ACHAT (entrée long) :
  - La moyenne mobile (SMA) est ascendante depuis 2 bougies
    (sma[t] > sma[t-1] > sma[t-2])
  - ET le volume est ascendant (volume[t] > volume[t-1])
  - ET FILTRE DE TENDANCE : le prix de clôture est au-dessus d'une SMA longue
    (TREND_MA_PERIOD), pour n'acheter que dans un marché haussier confirmé

VENTE (sortie long) :
  - La moyenne mobile "coupe" une bougie baissière : le prix de clôture est
    passé en-dessous de la SMA
  - ET il y a BEARISH_CANDLES_REQUIRED (3 par défaut) bougies baissières
    d'affilée, toutes sous la SMA — confirmation renforcée pour éviter de
    vendre sur du simple bruit (voir NOTE ci-dessous)

Hypothèses faites faute de précision (facilement modifiables ci-dessous) :
  - "moyenne mobile" = SMA simple, période 20 (MA_PERIOD)
  - "volume ascendant" = volume de la bougie en cours > volume de la bougie précédente
    (les données OHLCV de Freqtrade n'ont qu'une seule colonne "volume", pas de
    volume acheteur/vendeur séparé — c'est donc ce volume qui est utilisé)
  - timeframe : 5m

NOTE — historique des ajustements :
  1) Version initiale (2 bougies baissières, sans filtre de tendance) :
     -63.67% sur 2026-04-01 → 2026-07-18, exit_signal gagnant à 0.3%.
  2) Ajout du filtre de tendance (close > SMA longue) : -59.28%, amélioration
     marginale — le vrai problème n'était pas les entrées mais les sorties :
     503 sorties ROI gagnantes à 66.6% (+218 USDC), contre 1286 sorties
     exit_signal gagnantes à 0.4% (-2883 USDC). Le signal de vente d'origine
     confirmait quasi systématiquement une perte déjà entamée plutôt que de
     sécuriser un gain.
  3) Version actuelle : signal de vente rendu moins réactif (3 bougies
     baissières consécutives, toutes sous la SMA, au lieu de 2) pour filtrer
     le bruit du 5m. À rebacktester pour vérifier l'amélioration.

À tester en backtest avant tout usage en réel :
  freqtrade backtesting --strategy MMVolumeStrategy --timeframe 5m
"""

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class MMVolumeStrategy(IStrategy):

    INTERFACE_VERSION = 3

    # Période de la moyenne mobile (signal d'achat/vente)
    MA_PERIOD = 20

    # Période de la moyenne mobile longue (filtre de tendance)
    TREND_MA_PERIOD = 100

    # Nombre de bougies baissières consécutives requises pour vendre
    # (augmenté de 2 à 3 pour rendre le signal moins réactif au bruit)
    BEARISH_CANDLES_REQUIRED = 3

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

    startup_candle_count: int = max(MA_PERIOD, TREND_MA_PERIOD) + 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Moyenne mobile simple
        dataframe["sma"] = ta.SMA(dataframe, timeperiod=self.MA_PERIOD)

        # Moyenne mobile longue — filtre de tendance
        dataframe["sma_trend"] = ta.SMA(dataframe, timeperiod=self.TREND_MA_PERIOD)
        dataframe["uptrend"] = dataframe["close"] > dataframe["sma_trend"]

        # Bougie baissière = clôture sous l'ouverture
        dataframe["bearish_candle"] = dataframe["close"] < dataframe["open"]

        # SMA ascendante depuis 2 bougies : sma[t] > sma[t-1] > sma[t-2]
        dataframe["sma_rising_2"] = (
            (dataframe["sma"] > dataframe["sma"].shift(1))
            & (dataframe["sma"].shift(1) > dataframe["sma"].shift(2))
        )

        # Volume ascendant (bougie en cours > bougie précédente)
        dataframe["volume_rising"] = dataframe["volume"] > dataframe["volume"].shift(1)

        # Close en-dessous de la SMA (confirmation du "croisement")
        dataframe["below_sma"] = dataframe["close"] < dataframe["sma"]

        # N bougies baissières d'affilée, toutes sous la SMA (signal de vente
        # moins réactif que la version initiale — voir NOTE en tête de fichier)
        n = self.BEARISH_CANDLES_REQUIRED
        dataframe["bearish_streak"] = (
            dataframe["bearish_candle"].rolling(n).sum() == n
        )
        dataframe["below_sma_streak"] = (
            dataframe["below_sma"].rolling(n).sum() == n
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["sma_rising_2"]
                & dataframe["volume_rising"]
                & dataframe["uptrend"]
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["bearish_streak"]
                & dataframe["below_sma_streak"]
            ),
            "exit_long",
        ] = 1

        return dataframe

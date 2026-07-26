"""
MMVolumeStrategy — stratégie Freqtrade

LOGIQUE (telle que demandée, ajustée après backtests — voir NOTE ci-dessous) :

ACHAT (entrée long) :
  - La moyenne mobile (SMA) est ascendante depuis 2 bougies
    (sma[t] > sma[t-1] > sma[t-2])
  - ET le volume est ascendant (volume[t] > volume[t-1])
  - ET FILTRE DE TENDANCE : le prix de clôture est au-dessus d'une SMA longue
    (TREND_MA_PERIOD), pour n'acheter que dans un marché haussier confirmé

VENTE (sortie long) :
  - Uniquement via ROI (take-profit paliers) et stoploss — voir NOTE.

Hypothèses faites faute de précision (facilement modifiables ci-dessous) :
  - "moyenne mobile" = SMA simple, période 20 (MA_PERIOD)
  - "volume ascendant" = volume de la bougie en cours > volume de la bougie précédente
    (les données OHLCV de Freqtrade n'ont qu'une seule colonne "volume", pas de
    volume acheteur/vendeur séparé — c'est donc ce volume qui est utilisé)
  - timeframe : 5m

NOTE — historique des ajustements (backtests sur BTC/ETH/SOL/XRP-USDC,
2026-04-01 → 2026-07-18, marché en baisse de -11.82% sur la période) :
  1) Version initiale (achat SMA+volume, vente = croisement baissier + 2
     bougies baissières, sans filtre de tendance) : -63.67%, exit_signal
     gagnant à 0.3% (quasi jamais).
  2) Ajout du filtre de tendance à l'achat : -59.28% — amélioration marginale.
     Le problème n'était pas les entrées : 503 sorties ROI gagnantes à 66.6%
     (+218 USDC), contre 1286 sorties exit_signal gagnantes à 0.4% (-2883
     USDC).
  3) Signal de vente rendu moins réactif (3 bougies baissières au lieu de 2) :
     -60.72%, encore pire — exit_signal gagnant à 0% exact (0/1288).
  Conclusion : le signal de vente "MA coupe une bougie baissière + bougies
  baissières d'affilée" est structurellement un signal de CONFIRMATION de
  retournement, pas un signal prédictif — il ne peut que constater une perte
  après coup, jamais sécuriser un gain. Il a donc été retiré. Seules les
  sorties ROI (take-profit) et le stoploss gèrent maintenant les sorties.
  Le filtre de tendance sur les achats a été conservé.

  4) Backtest de cette version (ROI + stoploss -10%, sans signal de vente) :
     -13.02% — très proche du marché (-11.82%), cohérent pour une stratégie
     long-only sur marché baissier. Détail : 188 sorties ROI gagnantes à
     62.2% (+65 USDC), mais 22 sorties stoploss à -10.72% chacune (-608.91
     USDC) — le stoploss à -10% était trop large et a effacé le gain du ROI.
  5) Stoploss resserré de -10% à -5% pour limiter la casse par trade (voir
     valeur ci-dessous). À rebacktester pour confirmer l'amélioration.

  Il reste aussi à vérifier si la perte vient de la logique elle-même ou du
  fait que le marché était globalement baissier (-11.82%) sur cette période :
  tester sur une autre période/d'autres paires est recommandé.

À tester en backtest avant tout usage en réel :
  freqtrade backtesting --strategy MMVolumeStrategy --timeframe 5m
"""

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class MMVolumeStrategy(IStrategy):

    INTERFACE_VERSION = 3

    # Période de la moyenne mobile (signal d'achat)
    MA_PERIOD = 20

    # Période de la moyenne mobile longue (filtre de tendance)
    TREND_MA_PERIOD = 100

    # ROI / stoploss / timeframe — à ajuster selon votre profil de risque.
    # Ce sont désormais les SEULES sorties de la stratégie (pas de signal de
    # vente basé sur les bougies — voir NOTE en tête de fichier).
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0
    }
    stoploss = -0.05
    trailing_stop = False

    timeframe = "5m"

    # Pas de logique de sortie basée sur signal : ROI + stoploss uniquement
    use_exit_signal = False

    startup_candle_count: int = max(MA_PERIOD, TREND_MA_PERIOD) + 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Moyenne mobile simple
        dataframe["sma"] = ta.SMA(dataframe, timeperiod=self.MA_PERIOD)

        # Moyenne mobile longue — filtre de tendance
        dataframe["sma_trend"] = ta.SMA(dataframe, timeperiod=self.TREND_MA_PERIOD)
        dataframe["uptrend"] = dataframe["close"] > dataframe["sma_trend"]

        # SMA ascendante depuis 2 bougies : sma[t] > sma[t-1] > sma[t-2]
        dataframe["sma_rising_2"] = (
            (dataframe["sma"] > dataframe["sma"].shift(1))
            & (dataframe["sma"].shift(1) > dataframe["sma"].shift(2))
        )

        # Volume ascendant (bougie en cours > bougie précédente)
        dataframe["volume_rising"] = dataframe["volume"] > dataframe["volume"].shift(1)

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
        # Aucun signal de vente : sorties gérées uniquement par ROI/stoploss
        # (use_exit_signal = False ci-dessus). Méthode conservée car requise
        # par l'interface IStrategy.
        return dataframe

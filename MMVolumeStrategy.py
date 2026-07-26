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
  - Uniquement via ROI (take-profit paliers) et stoploss fixe — voir NOTE.

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
  5) Stoploss resserré à -5% : -24.93%, bien pire — 89 trades stoppés pour
     -1219.68 USDC (contre 22 trades et -608.91 USDC à -10%), pour un gain
     ROI à peine plus élevé (+116 USDC). Sur ce timeframe 5m, un stop fixe à
     -5% coupe trop de trades sur du simple bruit avant qu'ils n'atteignent
     le ROI.
  6) Stop dynamique basé sur l'ATR, multiplicateur 2.5x : catastrophique,
     -90.86%, 4750 trades (10x plus qu'avant), durée moyenne 45 minutes.
     L'ATR sur 5m ne représentait qu'une toute petite fraction du prix
     (0.1-0.3%), donc 2.5x ATR donnait un stop à ~0.3-0.8% au lieu des -10%
     attendus — trades stoppés quasi instantanément puis rachetés en boucle.
  7) Multiplicateur ATR augmenté à 20x : -26.96%, mieux que 2.5x (571 trades
     au lieu de 4750) mais toujours pire que le -10% fixe (-13.02%) — le stop
     moyen obtenu (-4.2%) restait plus serré et coupait encore trop de trades
     sur du bruit.
  Conclusion : sur ces paires et cette période, le stop fixe à -10% reste la
  meilleure configuration testée. Le stop dynamique ATR a été abandonné (code
  retiré) faute d'avantage démontré. Stoploss remis à -10%.

  8) Ajout d'un trailing stop par-dessus le stoploss -10%, activé à +3% de
     profit puis suivant le prix à 2% de distance : résultat rigoureusement
     identique au backtest sans trailing stop (-13.02%, mêmes 214 trades,
     aucune sortie "trailing_stop_loss"). Explication : le meilleur trade de
     tout le backtest ne dépassait que +2.63% de profit — jamais assez pour
     atteindre le seuil d'activation de +3%. Le trailing stop n'a donc jamais
     eu l'occasion de se déclencher.
  9) Seuils abaissés à +1.5% d'activation / 0.5% de distance, cohérents avec
     les gains réellement observés sur cette stratégie. À backtester.

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
    # Ce sont les SEULES sorties de la stratégie (pas de signal de vente basé
    # sur les bougies, pas de stop ATR — voir NOTE en tête de fichier).
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0
    }
    stoploss = -0.10

    # Trailing stop : s'active à +1.5% de profit, puis suit le prix à 0.5%
    # de distance. En dessous de +1.5%, seul le stoploss fixe -10% protège
    # le trade (trailing_only_offset_is_reached=True) — voir NOTE en tête de
    # fichier (seuil abaissé après un 1er essai à 3%/2% qui ne s'est jamais
    # déclenché : le meilleur trade du backtest ne dépassait que +2.63%).
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

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

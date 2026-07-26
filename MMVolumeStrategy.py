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
  5) Stoploss resserré à -5% : -24.93%, bien pire — 89 trades stoppés pour
     -1219.68 USDC (contre 22 trades et -608.91 USDC à -10%), pour un gain
     ROI à peine plus élevé (+116 USDC). Sur ce timeframe 5m, un stop fixe à
     -5% coupe trop de trades sur du simple bruit avant qu'ils n'atteignent
     le ROI. Un -10% universel n'est pas non plus idéal : trop large sur les
     paires calmes, potentiellement trop serré sur les paires volatiles.
  6) Stoploss remplacé par un stop dynamique basé sur l'ATR (volatilité
     récente de chaque paire) : distance = ATR_STOPLOSS_MULTIPLIER × ATR au
     moment de l'entrée, exprimée en % du prix d'entrée. Le stoploss fixe à
     -10% (STATIC_STOPLOSS_FLOOR) reste un filet de sécurité si l'ATR est
     indisponible. À backtester pour comparer au -10% fixe.

  Il reste aussi à vérifier si la perte vient de la logique elle-même ou du
  fait que le marché était globalement baissier (-11.82%) sur cette période :
  tester sur une autre période/d'autres paires est recommandé.

À tester en backtest avant tout usage en réel :
  freqtrade backtesting --strategy MMVolumeStrategy --timeframe 5m
"""

from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class MMVolumeStrategy(IStrategy):

    INTERFACE_VERSION = 3

    # Période de la moyenne mobile (signal d'achat)
    MA_PERIOD = 20

    # Période de la moyenne mobile longue (filtre de tendance)
    TREND_MA_PERIOD = 100

    # Période de l'ATR (mesure de volatilité) pour le stop dynamique
    ATR_PERIOD = 14

    # Distance du stop = ATR × ce multiplicateur, en % du prix d'entrée
    ATR_STOPLOSS_MULTIPLIER = 2.5

    # ROI / stoploss / timeframe — à ajuster selon votre profil de risque.
    # Le ROI et le stop (dynamique, voir custom_stoploss) sont désormais les
    # SEULES sorties de la stratégie (pas de signal de vente basé sur les
    # bougies — voir NOTE en tête de fichier).
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0
    }

    # Filet de sécurité si l'ATR est indisponible (ex: tout début de trade) —
    # c'est aussi la pire perte possible autorisée, custom_stoploss ne peut
    # jamais aller au-delà de cette valeur.
    stoploss = -0.10
    trailing_stop = False

    # Stop dynamique basé sur l'ATR (remplace le -10% fixe pour la plupart
    # des trades) — voir custom_stoploss ci-dessous et NOTE en tête de fichier
    use_custom_stoploss = True

    timeframe = "5m"

    # Pas de logique de sortie basée sur signal : ROI + stoploss uniquement
    use_exit_signal = False

    startup_candle_count: int = max(MA_PERIOD, TREND_MA_PERIOD, ATR_PERIOD) + 5

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

        # ATR — mesure de volatilité récente, utilisée par custom_stoploss
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.ATR_PERIOD)

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

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Stop dynamique basé sur l'ATR au moment de l'entrée : plus la paire
        est volatile à cet instant, plus le stop est large, et inversement.
        Ne bouge pas ensuite (pas un trailing stop) — juste une distance
        adaptée par trade au lieu d'un -10% universel.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        candles_before_entry = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        if candles_before_entry.empty:
            return self.stoploss

        atr_value = candles_before_entry.iloc[-1]["atr"]
        if atr_value is None or atr_value != atr_value or atr_value <= 0:
            # atr_value != atr_value détecte un NaN sans dépendance à numpy/pandas
            return self.stoploss

        stop_distance = (atr_value * self.ATR_STOPLOSS_MULTIPLIER) / trade.open_rate

        # Le stop dynamique ne doit jamais être plus large que le filet de
        # sécurité statique (self.stoploss est négatif, donc "plus large" =
        # plus négatif que lui)
        return max(-stop_distance, self.stoploss)

# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    CategoricalParameter,
    merge_informative_pair,
)
# --- Fin des libs ---


class remy(IStrategy):
    """
    Stratégie multi-indicateurs (RSI + EMA + volume) avec stoploss dynamique basé sur l'ATR.

    Historique : la version d'origine (backtest 380j, 5 paires) donnait 144 trades,
    9% de win rate, -6.89% de perf, même après 500 epochs de hyperopt sur
    buy/sell/protection avec min-trades=30. Verdict : pas un problème de réglage,
    la logique d'entrée n'avait pas d'edge. Diagnostic : le combo RSI-sort-de-survente
    + ADX déjà élevé + volume au-dessus de la moyenne tend à capter des pics
    d'exhaustion/retournement plutôt que des continuations propres (on achète
    des mèches, pas des respirations), et le stop ATR pouvait être si serré
    qu'il claquait sur du bruit normal.

    Modifications apportées :
      1. Filtre "pas de chasse au prix" : on n'entre que si le prix n'est pas
         déjà trop étiré au-dessus de l'EMA rapide (evite d'acheter au sommet
         d'un pic de volatilité). Nouveau paramètre : entry_max_ext_pct.
      2. ADX doit être EN HAUSSE sur les N dernières bougies, pas juste
         au-dessus d'un seuil statique : ça distingue une tendance qui se
         renforce d'une tendance déjà mature/en fin de course. Nouveau
         paramètre : adx_rising_periods.
      3. Plage hyperoptable du multiplicateur ATR élargie et remontée
         (1.5-6.0 au lieu de 1.0-4.0) pour laisser hyperopt trouver des stops
         moins nerveux si besoin.
      4. Plancher de distance de stop (min_stop_pct) : empêche un ATR
         anormalement bas (période calme) de produire un stop tellement
         serré qu'il se fait toucher par le bruit/spread normal.

    Logique (reste globalement identique) :
      - Filtre de tendance 15m : EMA rapide au-dessus de l'EMA lente => marché haussier
      - Filtre de tendance 1h  : prix au-dessus de son EMA 1h => tendance de fond haussière
      - Filtre de force        : ADX au-dessus d'un seuil ET en hausse => tendance qui se
                                 renforce (évite les faux signaux en marché plat ET les
                                 entrées tardives sur tendance déjà épuisée)
      - Anti-chasse            : le prix ne doit pas être trop loin au-dessus de l'EMA
                                 rapide au moment de l'entrée
      - Timing d'entrée        : RSI qui remonte depuis une zone de survente
      - Confirmation           : volume au-dessus de sa moyenne mobile
      - Sortie                 : RSI en zone de surachat OU retournement de tendance (croisement EMA)
      - Protection stoploss    : stoploss recalculé à chaque bougie en fonction de l'ATR,
                                 avec un plancher minimum de distance
      - Protections globales   : CooldownPeriod / MaxDrawdown / StoplossGuard pour couper le
                                 trading après une série de pertes ou un drawdown trop marqué

    Timeframe recommandé : 15m (avec confirmation 1h)
    """

    INTERFACE_VERSION = 3

    # --- Config générale ---
    timeframe = "15m"
    informative_timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Nombre de bougies nécessaires avant le premier signal exploitable.
    startup_candle_count: int = 210

    # --- ROI : paliers de sortie automatique en fonction du temps ---
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.025,
        "120": 0.0,
    }

    # --- Stoploss "de secours" ---
    stoploss = -0.10

    # --- Trailing stop simple (ignoré tant que use_custom_stoploss=True, gardé
    # comme filet si jamais custom_stoploss est désactivé) ---
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # --- Stoploss dynamique (custom_stoploss) ---
    use_custom_stoploss = True

    # --- Paramètres hyperoptimisables ---
    ema_fast_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    ema_slow_period = IntParameter(40, 100, default=50, space="buy", optimize=True)
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(25, 40, default=35, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(65, 80, default=70, space="sell", optimize=True)
    volume_factor = DecimalParameter(0.8, 2.0, default=1.0, decimals=1, space="buy", optimize=True)

    # Force de tendance (ADX) : filtre anti-marché plat
    adx_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    adx_threshold = IntParameter(15, 35, default=25, space="buy", optimize=True)
    # Nombre de bougies sur lesquelles l'ADX doit être en hausse continue
    # (tendance qui se renforce, pas déjà mature/en fin de course)
    adx_rising_periods = IntParameter(1, 5, default=2, space="buy", optimize=True)

    # Anti-chasse : écart maximum toléré (en %) entre le prix et l'EMA rapide
    # au moment de l'entrée. Empêche d'acheter en haut d'un pic de volatilité.
    entry_max_ext_pct = DecimalParameter(0.5, 5.0, default=2.0, decimals=1, space="buy", optimize=True)

    # EMA de la tendance de fond (timeframe informatif 1h)
    ema_trend_1h_period = IntParameter(20, 100, default=50, space="buy", optimize=True)

    atr_period = IntParameter(7, 21, default=14, space="protection", optimize=True)
    # Plage élargie et remontée : laisse hyperopt trouver des stops plus larges
    # si les valeurs serrées (proches de 1.0-2.0) se révèlent trop nerveuses.
    atr_stop_multiplier = DecimalParameter(1.5, 6.0, default=2.5, decimals=1, space="protection", optimize=True)
    atr_stop_multiplier_in_profit = DecimalParameter(0.8, 3.0, default=1.5, decimals=1, space="protection", optimize=True)
    # Plancher de distance de stop, en % du prix courant. Evite qu'un ATR
    # anormalement bas (période très calme) ne produise un stop irréaliste,
    # plus serré que le bruit/spread normal du marché.
    min_stop_pct = DecimalParameter(0.3, 2.0, default=0.6, decimals=1, space="protection", optimize=True)

    # Ordres au marché pour rester simple
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "orange"},
            "ema_slow": {"color": "blue"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "purple"},
            },
            "ADX": {
                "adx": {"color": "brown"},
            },
            "ATR": {
                "atr": {"color": "grey"},
            },
        },
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 4,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 2,
                "stop_duration_candles": 12,
                "only_per_pair": True,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 96,
                "trade_limit": 10,
                "stop_duration_candles": 24,
                "max_allowed_drawdown": 0.2,
            },
        ]

    def informative_pairs(self):
        pairs = self.dp.current_whitelist() if self.dp else []
        return [(pair, self.informative_timeframe) for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        indicators = {}

        # EMA rapide / lente : filtre de tendance
        for period in self.ema_fast_period.range:
            indicators[f"ema_fast_{period}"] = ta.EMA(dataframe, timeperiod=period)
        for period in self.ema_slow_period.range:
            indicators[f"ema_slow_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # RSI : timing d'entrée / sortie
        for period in self.rsi_period.range:
            indicators[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # ADX : force de la tendance (filtre anti-marché plat + anti-tendance-épuisée)
        for period in self.adx_period.range:
            indicators[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

        # ATR : base du stoploss dynamique
        for period in self.atr_period.range:
            indicators[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Injection groupée pour éviter la fragmentation du DataFrame
        dataframe = pd.concat([dataframe, pd.DataFrame(indicators, index=dataframe.index)], axis=1)

        dataframe["ema_fast"] = dataframe[f"ema_fast_{self.ema_fast_period.value}"]
        dataframe["ema_slow"] = dataframe[f"ema_slow_{self.ema_slow_period.value}"]
        dataframe["rsi"] = dataframe[f"rsi_{self.rsi_period.value}"]
        dataframe["adx"] = dataframe[f"adx_{self.adx_period.value}"]
        dataframe["atr"] = dataframe[f"atr_{self.atr_period.value}"]

        # Volume moyen : confirmation
        dataframe["volume_mean"] = dataframe["volume"].rolling(window=20).mean()

        # ADX en hausse continue sur N bougies : tendance qui se renforce,
        # pas juste "au-dessus d'un seuil" (qui peut être vrai en fin de tendance).
        adx_diff = dataframe["adx"].diff()
        adx_rising = pd.Series(True, index=dataframe.index)
        for shift in range(self.adx_rising_periods.value):
            adx_rising &= (adx_diff.shift(shift) > 0)
        dataframe["adx_rising"] = adx_rising.fillna(False)

        # Ecart du prix par rapport à l'EMA rapide, en % : sert de filtre anti-chasse
        # (on n'entre pas si le prix est déjà loin au-dessus de l'EMA, signe qu'on
        # arrive après le mouvement plutôt qu'au début).
        dataframe["ext_from_ema_pct"] = (
            (dataframe["close"] - dataframe["ema_fast"]) / dataframe["ema_fast"] * 100
        )

        # --- Tendance de fond en 1h (informative pair) ---
        if self.dp:
            informative = self.dp.get_pair_dataframe(
                pair=metadata["pair"], timeframe=self.informative_timeframe
            )
            if not informative.empty:
                informative[f"ema_trend_{self.informative_timeframe}"] = ta.EMA(
                    informative, timeperiod=self.ema_trend_1h_period.value
                )
                dataframe = merge_informative_pair(
                    dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True
                )
                dataframe["uptrend_1h"] = (
                    dataframe[f"close_{self.informative_timeframe}"]
                    > dataframe[f"ema_trend_{self.informative_timeframe}_{self.informative_timeframe}"]
                )
            else:
                dataframe["uptrend_1h"] = True  # pas de donnée dispo => ne bloque pas
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            # Tendance haussière 15m
            (dataframe["ema_fast"] > dataframe["ema_slow"]),
            # Tendance de fond haussière en 1h
            (dataframe["uptrend_1h"]),
            # Tendance suffisamment forte ET en train de se renforcer
            # (évite les marchés plats ET les tendances déjà mûres/épuisées)
            (dataframe["adx"] > self.adx_threshold.value),
            (dataframe["adx_rising"]),
            # Anti-chasse : le prix ne doit pas être trop étiré au-dessus de l'EMA rapide
            (dataframe["ext_from_ema_pct"] < self.entry_max_ext_pct.value),
            # RSI qui remonte depuis la zone de survente
            (qtpylib.crossed_above(dataframe["rsi"], self.rsi_buy_threshold.value)),
            # Confirmation par le volume
            (dataframe["volume"] > dataframe["volume_mean"] * self.volume_factor.value),
            (dataframe["volume"] > 0),
        ]
        dataframe.loc[
            np.logical_and.reduce(conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "rsi_recovery_uptrend_confirmed")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_rsi_exit = [
            (dataframe["rsi"] > self.rsi_sell_threshold.value),
            (dataframe["volume"] > 0),
        ]
        conditions_trend_exit = [
            (qtpylib.crossed_below(dataframe["ema_fast"], dataframe["ema_slow"])),
            (dataframe["volume"] > 0),
        ]
        dataframe.loc[
            np.logical_and.reduce(conditions_rsi_exit),
            ["exit_long", "exit_tag"],
        ] = (1, "rsi_overbought")
        dataframe.loc[
            np.logical_and.reduce(conditions_trend_exit),
            ["exit_long", "exit_tag"],
        ] = (1, "trend_reversal")
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Stoploss dynamique basé sur l'ATR, avec plancher minimum :
          - Hors profit : stop large = max(ATR * atr_stop_multiplier, min_stop_pct)
          - Une fois en profit : stop resserré = max(ATR * atr_stop_multiplier_in_profit, min_stop_pct)
            (protège les gains sans sortir trop tôt sur du bruit)

        Le plancher min_stop_pct évite qu'une période de très faible volatilité
        (ATR anormalement bas) ne produise un stop plus serré que le bruit/spread
        normal du marché, ce qui garantirait un stop-out quasi immédiat.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss
        last_candle = dataframe.iloc[-1]
        atr = last_candle.get("atr")
        if atr is None or pd.isna(atr) or atr <= 0 or current_rate <= 0:
            return self.stoploss

        if current_profit > 0.015:
            multiplier = self.atr_stop_multiplier_in_profit.value
        else:
            multiplier = self.atr_stop_multiplier.value

        atr_stop_distance_ratio = (atr * multiplier) / current_rate
        floor_ratio = self.min_stop_pct.value / 100

        stop_distance_ratio = max(atr_stop_distance_ratio, floor_ratio)

        # custom_stoploss doit retourner une valeur négative (fraction du prix actuel)
        return -abs(stop_distance_ratio)

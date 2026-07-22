#!/usr/bin/env python3
"""
Analyse les trades perdants d'un backtest freqtrade.

Lit le dernier résultat de backtest (ou un fichier précis passé en argument)
et sort une analyse détaillée des trades perdants : répartition par paire,
par heure d'entrée, par jour de semaine, durée, et liste des pires trades.

Usage:
    python3 analyze_losses.py
        (utilise automatiquement le dernier backtest via .last_result.json)
    python3 analyze_losses.py /freqtrade/user_data/backtest_results/backtest-result-XXXX.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_latest(backtest_dir: Path) -> Path:
    pointer = backtest_dir / ".last_result.json"
    if pointer.exists():
        with open(pointer) as f:
            data = json.load(f)
        latest = data.get("latest_backtest")
        if latest:
            return backtest_dir / latest
    # fallback: fichier le plus récent
    candidates = sorted(backtest_dir.glob("backtest-result-*.json"))
    candidates = [c for c in candidates if not c.name.endswith(".meta.json")]
    if not candidates:
        sys.exit(f"Aucun résultat de backtest trouvé dans {backtest_dir}")
    return candidates[-1]


def parse_iso(ts: str):
    from datetime import datetime
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def main():
    ap = argparse.ArgumentParser(description="Analyse les trades perdants d'un backtest freqtrade")
    ap.add_argument("result_file", nargs="?", help="Chemin vers le fichier backtest-result-*.json")
    ap.add_argument("--top", type=int, default=15, help="Nombre de pires trades à afficher (défaut 15)")
    args = ap.parse_args()

    if args.result_file:
        path = Path(args.result_file)
    else:
        path = find_latest(Path("/freqtrade/user_data/backtest_results"))

    print(f"Fichier analysé : {path}\n")
    data = load_result(path)

    strategies = data.get("strategy", {})
    if not strategies:
        sys.exit("Format inattendu : pas de clé 'strategy' dans le résultat.")

    strat_name = next(iter(strategies))
    trades = strategies[strat_name].get("trades", [])
    if not trades:
        sys.exit("Aucun trade dans ce résultat.")

    losers = [t for t in trades if t.get("profit_ratio", 0) < 0]
    winners = [t for t in trades if t.get("profit_ratio", 0) >= 0]

    print(f"Stratégie : {strat_name}")
    print(f"Total trades : {len(trades)}  |  Perdants : {len(losers)}  |  Gagnants : {len(winners)}\n")

    if not losers:
        print("Aucun trade perdant. Rien à analyser.")
        return

    # --- Par paire ---
    pair_counts = Counter(t["pair"] for t in losers)
    pair_totals = Counter(t["pair"] for t in trades)
    print("Perdants par paire (perdants / total trades sur la paire) :")
    for pair, cnt in pair_counts.most_common():
        total = pair_totals[pair]
        print(f"  {pair:<12} {cnt:>3} / {total:<3} ({100*cnt/total:5.1f}%)")

    # --- Par heure d'entrée (UTC) ---
    hour_counts = Counter(parse_iso(t["open_date"]).hour for t in losers)
    print("\nPerdants par heure d'entrée (UTC) :")
    for h in sorted(hour_counts):
        bar = "#" * hour_counts[h]
        print(f"  {h:02d}h  {hour_counts[h]:>3}  {bar}")

    # --- Par jour de semaine ---
    days_fr = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    weekday_counts = Counter(parse_iso(t["open_date"]).weekday() for t in losers)
    print("\nPerdants par jour de semaine :")
    for d in range(7):
        cnt = weekday_counts.get(d, 0)
        bar = "#" * cnt
        print(f"  {days_fr[d]:<10} {cnt:>3}  {bar}")

    # --- Durée ---
    def duration_min(t):
        return (parse_iso(t["close_date"]) - parse_iso(t["open_date"])).total_seconds() / 60

    loser_durations = [duration_min(t) for t in losers]
    winner_durations = [duration_min(t) for t in winners]
    print(f"\nDurée moyenne perdants : {sum(loser_durations)/len(loser_durations):.0f} min "
          f"(min {min(loser_durations):.0f} / max {max(loser_durations):.0f})")
    if winner_durations:
        print(f"Durée moyenne gagnants : {sum(winner_durations)/len(winner_durations):.0f} min "
              f"(min {min(winner_durations):.0f} / max {max(winner_durations):.0f})")

    # --- Motif de sortie ---
    exit_reasons = Counter(t.get("exit_reason", "?") for t in losers)
    print("\nMotif de sortie des perdants :")
    for reason, cnt in exit_reasons.most_common():
        print(f"  {reason:<20} {cnt}")

    # --- Enchaînements de pertes consécutives ---
    trades_sorted = sorted(trades, key=lambda t: t["open_date"])
    streak = 0
    max_streak = 0
    for t in trades_sorted:
        if t.get("profit_ratio", 0) < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"\nPlus longue série de pertes consécutives : {max_streak}")

    # --- Pires trades ---
    worst = sorted(losers, key=lambda t: t["profit_ratio"])[: args.top]
    print(f"\nLes {len(worst)} pires trades :")
    print(f"  {'Paire':<12} {'Entrée':<17} {'Durée':>6} {'Profit%':>8}  Motif")
    for t in worst:
        entry = parse_iso(t["open_date"]).strftime("%Y-%m-%d %H:%M")
        dur = duration_min(t)
        print(f"  {t['pair']:<12} {entry:<17} {dur:>5.0f}m {t['profit_ratio']*100:>7.2f}%  {t.get('exit_reason', '?')}")


if __name__ == "__main__":
    main()

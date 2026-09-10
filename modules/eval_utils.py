
# Funzioni condivise fra il notebook di training (01_training.ipynb) e quello
# di analisi (02_domande_di_ricerca.ipynb): simulazione di un episodio,
# metriche (errore di lux, energia/costo, jitter), grafico "giorno tipo" e
# wrapper per l'ambiente monitorato usato in training.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from lighting_env import AdaLux

LED_POWER_WATTS = 24.0            # potenza LED a piena potenza (ipotesi di progetto)
ENERGY_PRICE_EUR_PER_KWH = 0.25   # prezzo indicativo energia

colors_algo = {"PPO": "#dc2626", "SAC": "#2563eb"}


def energy_kwh(led_levels, step_minutes=AdaLux.STEP_MINUTES, power_watts=LED_POWER_WATTS):
    # energia consumata dai LED durante l'episodio
    hours_per_step = step_minutes / 60.0
    return float(sum(l * power_watts / 1000.0 * hours_per_step for l in led_levels))


def energy_cost_eur(kwh, price_eur_per_kwh=ENERGY_PRICE_EUR_PER_KWH):
    # costo economico dell'energia consumata, al prezzo indicativo sopra
    return kwh * price_eur_per_kwh


def actuator_jitter(records):
    # calcolo del movimento medio e massimo degli attuatori
    if len(records) < 2:
        return 0.0, 0.0
    deltas = [
        abs(records[i]["blind_pos"] - records[i - 1]["blind_pos"])
        + abs(records[i]["led_level"] - records[i - 1]["led_level"])
        for i in range(1, len(records))
    ]
    return float(np.mean(deltas)), float(np.max(deltas))


def run_episode(policy, seed, env_cls=AdaLux, is_baseline=False):
    # esegue un episodio sia per i modelli che per il controllore a soglie fisse
    env = env_cls(seed=seed)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    records = []
    while not done:
        if is_baseline:
            action, _ = policy.predict(obs, env=env)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        info["reward"] = reward
        records.append(info)
        done = terminated or truncated
    return records, env.lux_target, env.season_summer


def episode_metrics(records):
    # calcolo dell'errore medio di lux e di energia in un episodio
    occ = [r for r in records if r["occupied"]]
    mean_lux_err = float(np.mean([r["lux_error"] for r in occ])) if occ else 0.0
    mean_blind_pos_occ = float(np.mean([r["blind_pos"] for r in occ])) if occ else 0.0
    kwh = energy_kwh([r["led_level"] for r in records])
    j_mean, j_max = actuator_jitter(records)
    return {
        "mean_lux_error_pct": mean_lux_err * 100.0,
        "energy_kwh": kwh,
        "jitter_mean": j_mean,
        "jitter_max": j_max,
        "mean_blind_pos_occ": mean_blind_pos_occ,
    }


def evaluate_over_seeds(policy, seeds, env_cls=AdaLux, is_baseline=False,
                         label="policy"):
    # valutazione di una policy su piu episodi di test
    rows = []
    for seed in tqdm(seeds, desc=f"Valutazione {label}", leave=False):
        records, lux_target, season = run_episode(policy, seed, env_cls=env_cls,
                                                    is_baseline=is_baseline)
        m = episode_metrics(records)
        m.update({"seed": seed, "policy": label, "lux_target": lux_target,
                   "season_summer": season})
        rows.append(m)
    return pd.DataFrame(rows)


def plot_day(records, lux_target, season_summer, title_suffix=""):
    # grafico utilizzato per il controllo qualitativo di un episodio
    hours = [r["hour"] for r in records]
    indoor_lux = [r["indoor_lux"] for r in records]
    ext_lux = [r["ext_lux"] for r in records]
    blind = [r["blind_pos"] for r in records]
    led = [r["led_level"] for r in records]
    occ = [r["occupied"] for r in records]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax = axes[0]
    ax.plot(hours, indoor_lux, label="Lux interno", color="#d97706", linewidth=2)
    ax.plot(hours, ext_lux, label="Lux esterno (potenziale)", color="#94a3b8",
            linestyle="--", linewidth=1)
    ax.axhline(lux_target, color="#16a34a", linestyle=":", label="Target lux")
    _shade_occupancy(ax, hours, occ)
    ax.set_ylabel("Lux")
    ax.set_title(f"{'Estate' if season_summer else 'Inverno/mezza stagione'} — "
                 f"target {lux_target:.0f} lux {title_suffix}")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(hours, blind, label="Tapparella (0=chiusa,1=aperta)", color="#2563eb")
    ax.plot(hours, led, label="LED (0=spento,1=max)", color="#f59e0b")
    _shade_occupancy(ax, hours, occ)
    ax.set_ylabel("Livello [0-1]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Ora del giorno")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def _shade_occupancy(ax, hours, occ):
    start = None
    for i, o in enumerate(occ):
        if o and start is None:
            start = hours[i]
        if (not o or i == len(occ) - 1) and start is not None:
            end = hours[i]
            ax.axvspan(start, end, color="#22c55e", alpha=0.07)
            start = None


def make_monitored_env(seed, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    env = AdaLux(seed=seed)
    env = TimeLimit(env, max_episode_steps=AdaLux.STEPS_PER_DAY)
    env = Monitor(env, filename=os.path.join(log_dir, "monitor"))
    return env


def smoothed_reward_curve(log_dir, window=10):
    # carica i risultati dai log e applica una rolling mean al reward
    x, y = ts2xy(load_results(log_dir), "timesteps")
    if len(y) >= window:
        y_smooth = pd.Series(y).rolling(window, min_periods=1).mean().values
    else:
        y_smooth = y
    return x, y_smooth

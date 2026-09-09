"""
AdaLux
======
Ambiente Gymnasium che simula una stanza con:
  - una tapparella motorizzata (0 = chiusa, 1 = completamente aperta)
  - un impianto LED dimmerabile (0 = spento, 1 = massima potenza)

L'agente deve mantenere l'illuminamento (lux) sulla scrivania vicino a un
target (es. 500 lux per lettura/lavoro), sfruttando il più possibile la
luce naturale ed evitando l'abbagliamento (lux troppo sopra il target).

Un giorno simulato = 1 episodio, discretizzato a passi di 15 minuti
(96 passi). Stagione e nuvolosità cambiano ad ogni reset per generalizzare.

Stato (osservazione, tutto normalizzato in [-1, 1]):
    0: sin(ora del giorno)
    1: cos(ora del giorno)
    2: lux esterno normalizzato (potenziale lux sul piano di lavoro a
       tapparella 100% aperta)
    3: lux interno attuale normalizzato
    4: posizione tapparella attuale [0,1]
    5: livello LED attuale [0,1]
    6: lux target normalizzato
    7: stagione (0 = inverno/mezza stagione, 1 = estate)
    8: occupazione (0/1, 1 = c'è qualcuno che legge/lavora)

Azione: Box(2,) continua in [0,1]
    a[0] = posizione desiderata della tapparella (0..1)
    a[1] = livello desiderato dei LED (0..1)
Il movimento reale è limitato da una velocità massima per passo
(le tapparelle/i dimmer non saltano istantaneamente al target).

Reward (solo quando la stanza è occupata, altrimenti solo penalità energia):
    - errore di illuminamento rispetto al target (termine dominante)
    - penalità abbagliamento se lux interno supera una soglia
    - piccola penalità energetica proporzionale all'uso dei LED
    - piccola penalità di "jitter" per scoraggiare movimenti bruschi
    - bonus per la quota di illuminamento ottenuta da luce naturale,
      con un gate GRADUALE sull'errore di lux (pieno entro il 15% di
      errore, decresce linearmente fino ad annullarsi al 50%): dà un
      segnale anche a una policy non ancora accurata, invece di una
      soglia netta che azzera il bonus finché l'errore non è già basso
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AdaLux(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ----- costanti fisiche/di scenario -----
    STEPS_PER_DAY = 96          # 96 * 15 min = 24 h
    STEP_MINUTES = 15

    LUX_TARGET_READING = 500.0  # lux consigliati per lettura/lavoro (UNI EN 12464-1)
    LUX_TARGET_RANGE = (300.0, 750.0)   # variabilità del target tra episodi

    # Lux max potenziali sul PIANO DI LAVORO (non il cielo aperto) con tapparella
    # aperta e cielo sereno: un valore realistico per un ufficio/stanza con
    # finestra è nell'ordine di 3000-6000 lux nelle ore centrali della giornata.
    EXT_LUX_CLEAR_MAX = 5000.0
    LED_MAX_LUX = 850.0          # lux massimi erogabili dai LED a piena potenza

    GLARE_LUX_THRESHOLD_FACTOR = 1.6   # oltre 1.6x il target -> fastidio/abbagliamento
    GLARE_LUX_HARD_FACTOR = 2.5        # oltre 2.5x il target -> penalità forte

    MAX_BLIND_STEP = 0.40   # variazione massima frazione apertura per step (15 min)
    MAX_LED_STEP = 0.35     # variazione massima livello LED per step

    # gate graduale del daylight_bonus sull'errore di lux: bonus pieno entro
    # DAYLIGHT_GATE_FULL, decresce linearmente fino ad annullarsi oltre
    # DAYLIGHT_GATE_ZERO (vedi _compute_reward)
    DAYLIGHT_GATE_FULL = 0.15
    DAYLIGHT_GATE_ZERO = 0.50

    def __init__(self, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)

        # stato interno (inizializzato in reset)
        self.t = 0
        self.season_summer = False
        self.cloud_factor = 1.0
        self.indoor_lux = 0.0
        self.blind_pos = 0.0
        self.led_level = 0.0
        self.lux_target = self.LUX_TARGET_READING
        self.occupancy_start = 8
        self.occupancy_end = 19
        self._history = []

    # ------------------------------------------------------------------ #
    # Modello ambientale (sole, meteo)
    # ------------------------------------------------------------------ #
    def _hour(self) -> float:
        return (self.t * self.STEP_MINUTES / 60.0) % 24.0

    def _sun_elevation_factor(self, hour: float) -> float:
        """Fattore 0..1 che rappresenta l'altezza/intensità solare.
        Stagione estiva -> alba prima, tramonto dopo, picco più alto."""
        sunrise, sunset = (5.0, 21.0) if self.season_summer else (7.5, 17.5)
        if hour <= sunrise or hour >= sunset:
            return 0.0
        day_len = sunset - sunrise
        phase = (hour - sunrise) / day_len  # 0..1
        return float(np.sin(np.pi * phase)) ** 1.3  # picco a mezzogiorno solare

    def _external_lux(self, hour: float) -> float:
        sun = self._sun_elevation_factor(hour)
        base = sun * self.EXT_LUX_CLEAR_MAX
        return max(0.0, base * self.cloud_factor)

    def _update_cloud_factor(self):
        # random walk limitato in [0.15, 1.0] per simulare nuvolosità variabile
        step = self._rng.normal(0, 0.025)
        self.cloud_factor = float(np.clip(self.cloud_factor + step, 0.15, 1.0))

    def _is_occupied(self, hour: float) -> bool:
        return self.occupancy_start <= hour < self.occupancy_end

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0
        self.season_summer = bool(self._rng.random() < 0.5)
        self.cloud_factor = float(self._rng.uniform(0.4, 1.0))

        self.lux_target = float(self._rng.uniform(*self.LUX_TARGET_RANGE))
        self.occupancy_start = int(self._rng.integers(7, 10))
        self.occupancy_end = int(self._rng.integers(17, 20))

        self.blind_pos = float(self._rng.uniform(0.0, 0.3))
        self.led_level = 0.0
        self.indoor_lux = 0.0
        self._history = []

        obs = self._build_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        target_blind, target_led = float(action[0]), float(action[1])

        prev_blind, prev_led = self.blind_pos, self.led_level

        # attuatori con velocità limitata (fisicamente realistico)
        self.blind_pos += np.clip(target_blind - self.blind_pos,
                                   -self.MAX_BLIND_STEP, self.MAX_BLIND_STEP)
        self.led_level += np.clip(target_led - self.led_level,
                                   -self.MAX_LED_STEP, self.MAX_LED_STEP)
        self.blind_pos = float(np.clip(self.blind_pos, 0.0, 1.0))
        self.led_level = float(np.clip(self.led_level, 0.0, 1.0))

        hour = self._hour()
        self._update_cloud_factor()
        ext_lux = self._external_lux(hour)

        # --- illuminamento interno ---
        natural_component = self.blind_pos * ext_lux
        artificial_component = self.led_level * self.LED_MAX_LUX
        self.indoor_lux = natural_component + artificial_component

        occupied = self._is_occupied(hour)
        reward, reward_info = self._compute_reward(
            occupied, prev_blind, prev_led, ext_lux
        )

        self.t += 1
        terminated = False
        truncated = self.t >= self.STEPS_PER_DAY

        obs = self._build_obs()
        info = {
            "hour": hour,
            "occupied": occupied,
            "indoor_lux": self.indoor_lux,
            "ext_lux": ext_lux,
            "blind_pos": self.blind_pos,
            "led_level": self.led_level,
            **reward_info,
        }
        self._history.append(info)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def _compute_reward(self, occupied, prev_blind, prev_led, ext_lux):
        action_delta = abs(self.blind_pos - prev_blind) + abs(self.led_level - prev_led)
        smoothness_penalty = 0.05 * action_delta
        # penalità energetica dei LED: rende conveniente preferire la luce
        # naturale ai LED quando disponibile.
        energy_penalty = 0.35 * self.led_level

        if not occupied:
            # stanza vuota: risparmio energetico, nessun vincolo di lux stringente
            reward = -(energy_penalty + smoothness_penalty)
            return float(reward), {
                "lux_error": 0.0, "glare_penalty": 0.0, "daylight_bonus": 0.0,
            }

        # --- errore di illuminamento (termine dominante) ---
        lux_error = abs(self.indoor_lux - self.lux_target) / self.lux_target
        lux_error_penalty = 1.0 * min(lux_error, 3.0)

        # --- bonus esplicito per sfruttamento della luce naturale ---
        # gate GRADUALE sull'errore di lux (non più una soglia netta): pieno
        # bonus entro DAYLIGHT_GATE_FULL di errore, poi decresce linearmente
        # fino ad annullarsi oltre DAYLIGHT_GATE_ZERO. Una soglia netta
        # (bonus 0 finché lux_error >= 0.25, poi 0.5*natural_ratio) dava un
        # segnale nullo per l'esplorazione: se la tapparella resta chiusa
        # natural_component è sempre 0, quindi il bonus è sempre 0 finché
        # la policy non scopre "per caso" di aprire la tapparella mentre è
        # già accurata — un evento raro con esplorazione debole (es. PPO),
        # che infatti può restare bloccato su una soluzione LED-only. Il
        # gate graduale dà un segnale (per quanto piccolo) anche quando
        # l'errore non è ancora azzerato, incentivando l'esplorazione della
        # tapparella prima di aver già risolto il problema con i soli LED.
        natural_component = self.blind_pos * ext_lux
        if self.indoor_lux > 1e-3:
            natural_ratio = natural_component / self.indoor_lux
            error_gate = np.clip(
                1.0 - (lux_error - self.DAYLIGHT_GATE_FULL)
                / (self.DAYLIGHT_GATE_ZERO - self.DAYLIGHT_GATE_FULL),
                0.0, 1.0,
            )
            daylight_bonus = 0.5 * natural_ratio * error_gate
        else:
            daylight_bonus = 0.0

        # --- abbagliamento ---
        glare_soft = self.lux_target * self.GLARE_LUX_THRESHOLD_FACTOR
        glare_hard = self.lux_target * self.GLARE_LUX_HARD_FACTOR
        if self.indoor_lux <= glare_soft:
            glare_penalty = 0.0
        elif self.indoor_lux <= glare_hard:
            glare_penalty = 0.6 * (self.indoor_lux - glare_soft) / (glare_hard - glare_soft)
        else:
            glare_penalty = 0.6 + 1.4 * min(
                (self.indoor_lux - glare_hard) / glare_hard, 1.0
            )

        reward = (
            daylight_bonus
            - lux_error_penalty
            - glare_penalty
            - energy_penalty
            - smoothness_penalty
        )

        return float(reward), {
            "lux_error": lux_error,
            "glare_penalty": glare_penalty,
            "daylight_bonus": daylight_bonus,
        }

    # ------------------------------------------------------------------ #
    # Osservazione
    # ------------------------------------------------------------------ #
    def _build_obs(self) -> np.ndarray:
        hour = self._hour()
        ext_lux = self._external_lux(hour)
        occupied = self._is_occupied(hour)

        obs = np.array([
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.clip(ext_lux / self.EXT_LUX_CLEAR_MAX, 0.0, 1.0) * 2 - 1,
            np.clip(self.indoor_lux / (self.lux_target * 3.0), 0.0, 1.0) * 2 - 1,
            self.blind_pos * 2 - 1,
            self.led_level * 2 - 1,
            np.clip((self.lux_target - self.LUX_TARGET_RANGE[0]) /
                    (self.LUX_TARGET_RANGE[1] - self.LUX_TARGET_RANGE[0]), 0.0, 1.0) * 2 - 1,
            1.0 if self.season_summer else -1.0,
            1.0 if occupied else -1.0,
        ], dtype=np.float32)
        return obs

    def render(self):
        if not self._history:
            return
        h = self._history[-1]
        print(
            f"t={self.t:3d} h={h['hour']:5.2f} occ={int(h['occupied'])} "
            f"blind={h['blind_pos']:.2f} led={h['led_level']:.2f} "
            f"lux={h['indoor_lux']:7.1f}(target={self.lux_target:.0f})"
        )

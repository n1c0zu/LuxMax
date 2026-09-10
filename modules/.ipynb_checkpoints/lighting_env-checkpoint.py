from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AdaLux(gym.Env):
    metadata = {"render_modes": ["human"]}

    STEPS_PER_DAY = 96
    STEP_MINUTES = 15

    LUX_TARGET_READING = 500.0         # lux consigliati per lettura/lavoro (UNI EN 12464-1)
    LUX_TARGET_RANGE = (300.0, 750.0)  # variabilità del target tra episodi

    EXT_LUX_CLEAR_MAX = 5000.0   # Lux massimi provenienti dall'esterno
    LED_MAX_LUX = 850.0          # lux massimi erogabili dal LED

    GLARE_LUX_THRESHOLD_FACTOR = 1.6   # valore di soglia per l'abbagliamento (penalita leggera)
    GLARE_LUX_HARD_FACTOR = 2.5        # valore di soglia per l'abbagliamento (penalita forte)

    MAX_BLIND_STEP = 0.40   # variazione massima di apertura per step
    MAX_LED_STEP = 0.35     # variazione massima di luce erogata dal LED per step

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

        # stato interno
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

    
    # Modello ambientale di sole e meteo
    def _hour(self) -> float:
        return (self.t * self.STEP_MINUTES / 60.0) % 24.0

    def _sun_elevation_factor(self, hour: float) -> float:                    # fattore da 0 a 1 che indica l'altezza del sole
        sunrise, sunset = (5.0, 21.0) if self.season_summer else (7.5, 17.5)  # giornate più lunghe in estate
        if hour <= sunrise or hour >= sunset:
            return 0.0
        day_len = sunset - sunrise
        phase = (hour - sunrise) / day_len          # phase da 0 a 1
        return float(np.sin(np.pi * phase)) ** 1.3  # picco a mezzogiorno

    def _external_lux(self, hour: float) -> float: # calcolo del lux esterno utilizzando intesità solare e interferenza delle nuvole
        sun = self._sun_elevation_factor(hour)
        base = sun * self.EXT_LUX_CLEAR_MAX
        return max(0.0, base * self.cloud_factor)

    def _update_cloud_factor(self):
        # numero limitato tra 0.15 e 1.0 per simulare nuvolosità variabile
        step = self._rng.normal(0, 0.025)
        self.cloud_factor = float(np.clip(self.cloud_factor + step, 0.15, 1.0))

    def _is_occupied(self, hour: float) -> bool:
        return self.occupancy_start <= hour < self.occupancy_end

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # inizializzazione dei valori
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0
        self.season_summer = bool(self._rng.random() < 0.5)
        self.cloud_factor = float(self._rng.uniform(0.15, 1.0))

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

        # attuatori con velocità limitata
        self.blind_pos += np.clip(target_blind - self.blind_pos,
                                   -self.MAX_BLIND_STEP, self.MAX_BLIND_STEP)
        self.led_level += np.clip(target_led - self.led_level,
                                   -self.MAX_LED_STEP, self.MAX_LED_STEP)
        self.blind_pos = float(np.clip(self.blind_pos, 0.0, 1.0))
        self.led_level = float(np.clip(self.led_level, 0.0, 1.0))

        hour = self._hour()
        self._update_cloud_factor()
        ext_lux = self._external_lux(hour)

        # illuminamento interno
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

    # Reward
    def _compute_reward(self, occupied, prev_blind, prev_led, ext_lux):
        action_delta = abs(self.blind_pos - prev_blind) + abs(self.led_level - prev_led)
        smoothness_penalty = 0.05 * action_delta
        # penalità energetica del LED
        energy_penalty = 0.35 * self.led_level

        if not occupied:
            # se la stanza è vuota non si hanno penalità di target di luce ne di comfort
            reward = -(energy_penalty + smoothness_penalty)
            return float(reward), {
                "lux_error": 0.0, "glare_penalty": 0.0, "daylight_bonus": 0.0,
            }

        # errore di illuminamento
        lux_error = abs(self.indoor_lux - self.lux_target) / self.lux_target
        lux_error_penalty = 1.0 * min(lux_error, 3.0)

        # Calcola il daylight_bonus per lo sfruttamento della luce naturale.
        # Il bonus è proporzionale al rapporto tra la luce naturale stimata e l'illuminazione
        # interna totale, e viene modulato dall'error_gate basato sull'errore 
        # rispetto al target (lux_error). In assenza di luce interna, il bonus viene azzerato.
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

        # abbagliamento
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

    # Osservazione
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

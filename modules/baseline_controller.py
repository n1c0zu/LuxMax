from __future__ import annotations

import numpy as np

class ThresholdController:
    #Controllore a soglie fisse, tarato su un setpoint nominale

    def __init__(self, nominal_target: float = 500.0, deadband: float = 0.10):
        # setpoint
        self.nominal_target = nominal_target
        # deadband: soglia entro la quale il led non cambia stato
        self.deadband = deadband

    def reset(self):
        # Wrapper
        pass

    def _blind_from_lux(self, lux: float) -> float:
        t = self.nominal_target
        if lux > 2.5 * t:
            return 0.0
        elif lux > 1.6 * t:
            return 0.3
        elif lux > 1.0 * t:
            return 0.6
        else:
            return 1.0

    def act(self, env) -> np.ndarray:
        # Calcola l'azione [blind, led] leggendo lo stato fisico dell'env
        hour = env._hour()
        occupied = env._is_occupied(hour)

        if not occupied:
            return np.array([0.0, 0.0], dtype=np.float32)

        lux = env.indoor_lux
        t = self.nominal_target
        error = (lux - t) / t

        blind_action = self._blind_from_lux(lux)

        if error < -self.deadband:
            led_action = 1.0
        elif error > self.deadband:
            led_action = 0.0
        else:
            led_action = env.led_level  # dentro la banda morta non c'è nessuna variazione

        return np.array([blind_action, led_action], dtype=np.float32)

    def predict(self, obs, env=None, deterministic: bool = True):
        if env is None:
            raise ValueError("ThresholdController.predict richiede l'env (stato fisico reale).")
        action = self.act(env)
        return action, None

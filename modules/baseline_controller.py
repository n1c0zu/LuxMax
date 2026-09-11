from __future__ import annotations

import numpy as np

class ThresholdController:
    #Controllore a soglie fisse (regole if/else, nessun apprendimento).
    #Il setpoint non e' un valore nominale fisso: viene letto una sola volta
    #da env.lux_target all'inizio dell'episodio (reset()) e resta costante
    #per tutta la sua durata, come un termostato a cui si imposta una
    #soglia una volta sola all'avvio (non lo rilegge step per step).

    def __init__(self, nominal_target: float = 500.0, deadband: float = 0.10):
        # nominal_target: usato solo come fallback se reset() viene
        # chiamato senza passare l'env (es. uso standalone del controllore)
        self.nominal_target = nominal_target
        # target effettivo dell'episodio corrente, impostato da reset(env)
        self.target = nominal_target
        # deadband: soglia entro la quale il led non cambia stato
        self.deadband = deadband

    def reset(self, env=None):
        # Cattura il target dell'episodio UNA VOLTA SOLA a inizio episodio
        # (come un vero controllore a cui viene impostato il setpoint
        # all'avvio), invece di rileggerlo dall'env ad ogni step.
        self.target = env.lux_target if env is not None else self.nominal_target

    def _blind_from_lux(self, lux: float) -> float:
        t = self.target
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
        t = self.target
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

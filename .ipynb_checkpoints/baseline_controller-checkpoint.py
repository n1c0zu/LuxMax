"""
ThresholdController
====================
Controllore "classico" a soglie fisse (regole if/else, nessun apprendimento),
rappresentativo di una logica di building-automation tradizionale. Serve
come baseline di confronto per l'agente RL (Q1).

Punto chiave: un controllore a soglie fisse reale viene tarato UNA VOLTA in
fase di installazione su un valore di setpoint nominale (es. 500 lux, il
valore raccomandato per lettura/lavoro) e NON riceve come input il target
di lux richiesto in un dato momento: le sue soglie restano identiche in
ogni episodio, a differenza dell'agente RL che osserva il target corrente
(oss. indice 6) e può adattarsi episodio per episodio. Questo è uno dei
vantaggi strutturali dell'approccio RL che Q1 vuole mettere in evidenza.

Logica (fissa, non adattiva):
  - Stanza NON occupata -> tapparella chiusa, LED spenti (risparmio energetico).
  - Stanza occupata:
      - Tapparella: posizione scelta tra un piccolo numero di livelli fissi
        in base a soglie di lux assolute (multi-step, come un dimmer/tenda
        a più posizioni tipico di un BMS semplice) calcolate sul setpoint
        NOMINALE (non sul target reale dell'episodio).
      - LED: controllo bang-bang con banda morta (hysteresis) attorno al
        setpoint nominale -> se il lux è sotto il target di oltre
        `deadband` i LED vanno al massimo, se è sopra vanno a zero,
        altrimenti restano invariati (per non oscillare in continuazione
        dentro la banda).

Espone un'interfaccia `predict(obs, env)` compatibile con il resto del
codice di valutazione (evaluate.py / notebook), analoga a quella di un
modello Stable-Baselines3 (`model.predict(obs)`), ma richiede anche
l'ambiente per leggere lo stato fisico reale (lux, non solo l'osservazione
normalizzata), esattamente come farebbe un vero controllore che legge un
sensore di lux.
"""
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

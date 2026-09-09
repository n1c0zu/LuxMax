# AdaLux — Controllo Adattivo dell'Illuminazione (Tapparelle + LED) con RL

Agente di Reinforcement Learning che impara a bilanciare **luce naturale**
(apertura tapparella) e **luce artificiale** (dimmer LED) per mantenere un
illuminamento costante sul piano di lettura/lavoro, sfruttando il più
possibile la luce naturale ed evitando l'**abbagliamento**. L'ambiente
Gymnasium custom si chiama **AdaLux** (classe `AdaLux` in `lighting_env.py`).

## Algoritmo consigliato: SAC

Allenato con **PPO**, l'agente collassava su una soluzione sub-ottimale
(tapparella sempre chiusa, illuminamento gestito solo dai LED). **SAC**
(Soft Actor-Critic), grazie alla massimizzazione esplicita dell'entropia e
all'apprendimento off-policy, ha risolto il problema: con il modello
incluso l'errore medio di illuminamento nelle ore occupate è circa il
3-5%. Il confronto PPO vs SAC (e l'osservazione di questo comportamento)
è riprodotto nella sezione Q2 del notebook.

## Struttura del progetto

```
lighting_env.py            # ambiente Gymnasium custom AdaLux (simulazione fisica semplificata)
baseline_controller.py     # ThresholdController: baseline a soglie fisse (non-RL), usata come confronto per l'agente RL
notebook_lighting_rl.ipynb # training + valutazione end-to-end (Q1/Q2/Q3), vedi sotto
requirements.txt
models/sac_q1.zip           # modello SAC di Q1 (generato dal notebook)
models/ppo_q2.zip           # modello PPO di Q2 (generato dal notebook)
models/sac_q2.zip           # modello SAC di Q2 (generato dal notebook)
```

## Installazione

```bash
pip install -r requirements.txt
```

## Come funziona l'ambiente (`lighting_env.py` — classe `AdaLux`)

- **Episodio** = un giorno simulato, 96 step da 15 minuti.
- **Osservazione** (9 valori, normalizzati): ora del giorno (sin/cos), lux
  esterno potenziale, lux interno attuale, posizione tapparella, livello
  LED, lux target, stagione (estate/altro), occupazione della stanza.
- **Azione** continua `Box(2,)` in `[0,1]`: posizione desiderata della
  tapparella e livello desiderato dei LED. Gli attuatori si muovono con
  velocità limitata per step (comportamento realistico, non teletrasporto).
- **Dinamica**:
  - il sole segue un profilo stagionale (alba/tramonto, intensità);
  - `EXT_LUX_CLEAR_MAX = 5000` lux rappresenta il lux potenziale realistico
    sul **piano di lavoro** (non il cielo aperto) a tapparella aperta;
  - la nuvolosità varia come random walk (giornate non tutte uguali);
  - l'illuminamento interno = `tapparella × lux_esterno + LED × lux_LED_max`.
- **Reward** (attivo quando la stanza è occupata):
  - penalità principale sull'errore rispetto al target di lux;
  - **bonus esplicito** proporzionale alla frazione di illuminamento che
    proviene dalla tapparella, con un gate graduale sull'errore di lux
    (pieno entro il 15% di errore, si annulla oltre il 50%) invece di una
    soglia netta — dà un segnale anche a una policy non ancora accurata,
    incentivando l'esplorazione della tapparella prima di aver già
    risolto il problema con i soli LED;
  - penalità abbagliamento se il lux supera ~1.6-2.5× il target;
  - penalità energetica sull'uso dei LED (rende conveniente preferire la
    luce naturale);
  - piccola penalità di "jitter" per scoraggiare movimenti bruschi e
    continui di tapparella/LED.
- Target di lux, stagione, nuvolosità e fasce di occupazione cambiano ad
  ogni `reset()`, così l'agente generalizza invece di imparare un'unica
  routine fissa.

## Baseline non-RL (`baseline_controller.py`)

`ThresholdController` è un controllore a regole fisse (nessun
apprendimento), tarato su un setpoint nominale fisso di 500 lux e ignaro
del target reale dell'episodio — rappresenta la tipica logica di building
automation tradizionale. Serve da termine di paragone per l'agente RL
(domanda Q1 del notebook).

## Training e valutazione: `notebook_lighting_rl.ipynb`

Tutto il flusso di training e valutazione è nel notebook, organizzato per
rispondere a tre domande di ricerca:

- **Q1** — Un agente RL (SAC) supera il controllore a soglie fisse?
  Confronto su errore medio di lux ed energia consumata su un set di
  episodi di test.
- **Q2** — Quale tra PPO e SAC converge meglio ed evita ottimi locali?
  Training a parità di step, confronto delle curve di reward e ispezione
  qualitativa del comportamento appreso.
- **Q3** — La policy resta affidabile fuori distribuzione? Test su
  episodi con condizioni estreme (target di lux, nuvolosità, occupazione)
  non viste in training.

Il notebook ha una variabile `FAST_MODE` in cima: `True` esegue una
passata veloce (pochi timestep) per verificare che tutto funzioni, `False`
usa i timestep "di ricerca" consigliati (150.000 per SAC/PPO — più lenti,
soprattutto SAC che è single-env). C'è anche una variabile `DEVICE`
(`"auto"` / `"cuda"` / `"cpu"`) per scegliere GPU o CPU: con reti così
piccole la CPU è spesso comparabile o più veloce della GPU.

## Idee per estenderlo

- **Più training**: aumentare `TIMESTEPS_Q1`/`TIMESTEPS_Q2` nel
  notebook (con `FAST_MODE = False`) per una convergenza ancora più
  stabile.
- **Osservazioni reali**: sostituire i segnali simulati (lux esterno,
  nuvolosità) con letture da sensori reali (fotocellula da esterno,
  sensore di presenza) mantenendo la stessa struttura dell'osservazione.
- **Multi-stanza**: vettorizzare più istanze dell'ambiente con parametri
  diversi (orientamento finestra, superficie vetrata) per un agente che
  generalizza su più ambienti dello stesso edificio.
- **Domain randomization** più spinta (efficienza tapparella, potenza LED)
  per robustezza al trasferimento sim-to-real.
- **Deployment**: una volta addestrato, `model.predict(obs)` gira in pochi
  millisecondi su CPU: può essere eseguito periodicamente (es. ogni 5-15
  min) da un piccolo servizio/home-automation hub (es. Home Assistant) che
  legge i sensori, costruisce l'osservazione e invia i comandi a
  tapparella/dimmer.

# AdaLux — Controllo Adattivo dell'Illuminazione (Tapparelle + LED) con RL

Agente di Reinforcement Learning che impara a bilanciare **luce naturale**
(apertura tapparella) e **luce artificiale** (dimmer LED) per mantenere un
illuminamento costante sul piano di lettura/lavoro, sfruttando il più
possibile la luce naturale ed evitando l'**abbagliamento**. L'ambiente
Gymnasium custom si chiama **AdaLux** (classe `AdaLux` in
`modules/lighting_env.py`).

Il progetto risponde a tre domande di ricerca, riprodotte nel notebook di
analisi:

- **Q1** — Un agente RL (SAC) supera un controllore a soglie fisse?
- **Q2** — Quale tra PPO e SAC converge meglio ed evita ottimi locali?
- **Q3** — La policy resta affidabile fuori distribuzione (condizioni
  estreme non viste in training)?

## Struttura delle cartelle

```
modules/                        # codice condiviso, importato dai notebook
  lighting_env.py                #   ambiente Gymnasium custom AdaLux (simulazione fisica semplificata)
  baseline_controller.py         #   ThresholdController: baseline a soglie fisse (non-RL)
  eval_utils.py                  #   funzioni comuni: esecuzione episodi, metriche, grafico "giorno tipo"

notebooks/                      # notebook eseguibili, vedi "Ordine di esecuzione" sotto
  01_training.ipynb              #   allena PPO e SAC e salva i modelli
  02_domande_di_ricerca.ipynb    #   carica i modelli allenati e risponde a Q1/Q2/Q3

models/                         # modelli allenati (prodotti da 01_training.ipynb)
  ppo_q2_seed{1..N}.zip           #   un modello PPO per ogni training seed
  sac_q2_seed{1..N}.zip           #   un modello SAC per ogni training seed (riusato anche da Q1/Q3)

logs/                            # log di training per episodio (prodotti da 01_training.ipynb)
  q2_ppo_seed{1..N}/monitor.monitor.csv
  q2_sac_seed{1..N}/monitor.monitor.csv

figures/                         # grafici salvati in PNG (prodotti da 02_domande_di_ricerca.ipynb)

requirements.txt
README.md
```

## Dati e ambiente utilizzato

AdaLux genera gli episodi (giornate simulate) **proceduralmente**, con
un generatore di numeri casuali seedato in modo indipendente ad ogni
`reset()` — un seed diverso per ogni episodio, sia in training (`SEEDS`,
valori piccoli 1..N) sia in valutazione (`TEST_SEEDS`, valori a partire
da 1000, così da non sovrapporsi mai ai seed di training). Gli artefatti
salvati su disco sono quelli prodotti dall'esecuzione dei notebook: i
modelli allenati (`models/`), i log di reward per episodio (`logs/`) e le
figure generate in fase di analisi (`figures/`) — vedi la sezione
"Struttura delle cartelle" sopra per dove trovarli.

### Come funziona l'ambiente (`modules/lighting_env.py` — classe `AdaLux`)

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

### Baseline non-RL (`modules/baseline_controller.py`)

`ThresholdController` è un controllore a regole fisse (nessun
apprendimento), tarato su un setpoint nominale fisso di 500 lux e ignaro
del target reale dell'episodio — rappresenta la tipica logica di building
automation tradizionale. Serve da termine di paragone per l'agente RL
(domanda Q1).

## Librerie richieste

```bash
pip install -r requirements.txt
```

Le librerie principali sono `gymnasium` (ambiente), `stable-baselines3` e
`torch` (algoritmi PPO/SAC), `pandas`/`numpy` (metriche) e `matplotlib`
(grafici); l'elenco completo con le versioni minime è in
`requirements.txt`.

## Ordine di esecuzione dei notebook

Non c'è un notebook di preprocessing (vedi sopra: non c'è un dataset da
preparare). L'ordine è:

1. **`notebooks/01_training.ipynb`** — allena PPO e SAC su `N_SEEDS`
   training seed indipendenti e salva i modelli in `models/` e i log in
   `logs/`. Se i modelli sono già presenti su disco non riallena da zero
   (`RETRAIN = False` di default): puoi saltare questo notebook se
   `models/` è già popolata.
2. **`notebooks/02_domande_di_ricerca.ipynb`** — carica i modelli allenati
   da `models/` e risponde a Q1, Q2 e Q3, salvando i grafici in
   `figures/`. Va eseguito **dopo** il notebook di training, perché si
   aspetta di trovare i file `.zip` in `models/`.

**Importante**: in cima a entrambi i notebook ci sono le stesse variabili
di configurazione (`FAST_MODE`, `N_SEEDS`, `TIMESTEPS`,
`N_TEST_EPISODES`) — devono avere **lo stesso valore** in entrambi,
altrimenti il secondo notebook cerca modelli con un numero di seed
diverso da quelli effettivamente allenati dal primo. `FAST_MODE = True`
esegue una passata veloce (pochi timestep) solo per verificare che tutto
funzioni; `False` usa i timestep "di ricerca" consigliati (80.000 — più
lenti, soprattutto SAC che è single-env). `N_SEEDS` imposta su quanti
training seed indipendenti allenare PPO e SAC. `DEVICE` è impostato su
`"cpu"` (consigliato: con reti così piccole la CPU è spesso comparabile o
più veloce di una GPU); il notebook segnala solo a scopo informativo se
è disponibile una GPU CUDA.

## Idee per estenderlo

- **Più training**: aumentare `TIMESTEPS` nel notebook di training (con
  `FAST_MODE = False`) per una convergenza ancora più stabile.
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

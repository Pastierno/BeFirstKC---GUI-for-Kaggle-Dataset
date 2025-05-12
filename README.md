# BeFirstKC

Questo progetto nasce come esercizio di apprendimento per uno junior Python developer con un interesse crescente verso la **Data Analysis** e **Machine Learning**. L'obiettivo è quello di combinare analisi dei dati con un'interfaccia grafica interattiva costruita con **PyQt5**, esplorando concetti fondamentali come:
    - Manipolazione dei dati con Python (lato backend),
    - Presentazione di elementi visivi interattivi (lato frontend),
    - Gestione di animazioni e feedback visivi per l'utente.


**_Il tuo primo alleato per affrontare le Kaggle Competition con facilità_**

---

## Introduzione

BeFirstKC è un’app desktop basata su PyQt5 che guida gli utenti, anche alle prime armi, attraverso l’intero workflow di una competizione Kaggle:

1. **Caricamento** dei dati  
2. **Preprocessing** e gestione dei valori mancanti  
3. **Imputazione** delle feature categoriche  
4. **Encoding** delle variabili  
5. **Visualizzazione** dei dati  
6. **Addestramento** e **ottimizzazione** di modelli XGBoost / LightGBM  
7. **Generazione** del file di submission

Ti offre un’interfaccia grafica intuitiva senza dover scrivere alle prime armi lunghe righe di codice. Pronto a salire sul podio?

---

## Caratteristiche principali

- **Interfaccia a schede** divisa per fasi: Load, Preprocess, Impute Cat, Encode, Visualize, Model, Submission  
- **Preview** interattivo dei dataset (head e statistiche)  
- **Trasformazioni numeriche**: Yeo–Johnson, Standard Scaling  
- **Imputazione**: media, mediana, moda, strategia categoriale (mode, costante, random)  
- **Label & One-Hot Encoding**, con gestione automatica delle colonne  
- **Visualizzazioni**: istogrammi, boxplot, matrici di correlazione  
- **Modeling**: training rapido, valutazioni (accuracy/R², MAE, MSE, RMSE, confusion matrix)  
- **Hyperparameter tuning** con Optuna (ricerca automatica dei migliori parametri)  
- **Esportazione** del modello e del CSV di submission pronto per il caricamento su Kaggle  

---

## 🗂 Struttura del progetto

```
BeFirstKC/
├── main.py               # Entry point dell’applicazione
├── widgets.py            # Pulsanti animati e componenti custom
├── style.qss             # Foglio di stile Qt per temi e colori
├── fonts/                # Cartella con font (es. Roboto)
│   └── Roboto-Regular.ttf
├── README.md             # Questo file
└── requirements.txt      # Dipendenze Python
```

---

## Prerequisiti

- Python 3.7 - 3.10
- Sistema operativo: Windows, macOS o Linux  
- Pacchetti elencati in `requirements.txt` (PyQt5, pandas, scikit-learn, XGBoost, LightGBM, Optuna, seaborn, matplotlib…)

---

## Installazione

1. **Clona** il repository:
   ```bash
   git clone https://github.com/tuo-utente/BeFirstKC.git
   cd BeFirstKC
   ```
2. **Crea** un ambiente virtuale:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux / macOS
   venv\Scripts\activate     # Windows
   ```
3. **Installa** le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Avvio dell’app

```bash
python main.py
```

Alla partenza verrà caricata l’interfaccia grafica `BeFirstKC`. Se non trovi il font Roboto, l’app userà quello di sistema senza interrompersi.

---

## Workflow consigliato

1. **Load Data**  
   - Carica i file CSV di **train** e **test**.  
   - Visualizza le prime righe e le statistiche descrittive.

2. **Preprocess**  
   - Seleziona colonne numeriche da trasformare (Yeo–Johnson o Standard Scaling).  
   - Gestisci outlier, NaN e valori infiniti.  
   - Usa “Drop NaN” o “Impute Numeric” (media/mediana/moda).

3. **Impute Cat**  
   - Scegli le colonne categoriche e la strategia (Mode, Constant, Random).  
   - Esegui l’imputazione con un click.

4. **Encode**  
   - Attiva Label Encoding e/o One-Hot Encoding.  
   - Il tool rinomina e concatena automaticamente le nuove colonne.

5. **Visualize**  
   - Esplora la distribuzione delle variabili (istogrammi, boxplot).  
   - Studia le correlazioni con la heatmap.

6. **Model**  
   - Seleziona le colonne da droppare (es. ID o fuori tema).  
   - Scegli **Target** e tipo di modello (Classifier/Regressor).  
   - Esegui il **Train** veloce per un feedback istantaneo.  
   - Premi **Optimize** per far girare Optuna e trovare i parametri ottimali.  
   - Usa “Train Best Params” e confronta le metriche.

7. **Submission**  
   - Scegli la colonna ID e genera il CSV di submission.  
   - Carica il file su Kaggle e… buona competizione!

---

## Spiegazione dei concetti

- **Yeo–Johnson**: trasforma la distribuzione verso una forma più gaussiana, migliorando la performance di molti modelli.  
- **Standard Scaling**: normalizza le feature per avere media 0 e varianza 1, cruciale per algoritmi basati su gradiente.  
- **Label vs One-Hot Encoding**:  
  - *Label Encoding*: assegna interi a categorie—veloce ma può introdurre ordine artificiale.  
  - *One-Hot*: crea nuove colonne binarie, evita ordini ma aumenta dimensionalità.  
- **Optuna**: framework di hyperparameter tuning basato su ricerca bayesiana, che riduce i tempi di sperimentazione rispetto alla grid/random search.

---

## Consigli
  
- Monitora la **distribuzione** delle feature dopo le trasformazioni.  
- Occhio al **data leakage**: non applicare informazioni del test nel preprocessing del train.  
- Sperimenta diverse **metriche** a seconda del tipo di competizione (es. RMSE per regression, AUC per classification).

---


## Licenza

Distributed under the MIT License. Vedi `LICENSE` per i dettagli.

--- 
## Autori

## Autori

- **Marco**
[Linkedin](https://www.linkedin.com/in/marco-patierno-a933a6352/) | [Mail](marcopatierno.m@gmail.com)
- **Giovanni**
[Linkedin](https://www.linkedin.com/in/giovanni-pisaniello-094201317/) | [Mail](pisaniellogiovanni53@gmail.com)
- **Nunzio**
[Linkedin](https://www.linkedin.com/in/nunzio-de-cicco/) | [Mail](decicconunzio@gmail.com)

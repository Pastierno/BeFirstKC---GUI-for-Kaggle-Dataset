# Animated Glow Button in PyQt5

Questo progetto nasce come esercizio di apprendimento per un junior Python developer con un interesse crescente verso la **Data Analysis** e **Machine Learning**. L'obiettivo è quello di combinare analisi dei dati con un'interfaccia grafica interattiva costruita con **PyQt5**, esplorando concetti fondamentali come:
    - Manipolazione dei dati con Python (lato backend),
    - Presentazione di elementi visivi interattivi (lato frontend),
    - Gestione di animazioni e feedback visivi per l'utente.

## Indice

- [Funzionalità](#funzionalit%C3%A0)
- [Requisiti](#requisiti)
- [Installazione e Avvio](#installazione-e-avvio)
- [Struttura del Codice](#struttura-del-codice)
- [Anteprima](#anteprima)
- [Licenza](#licenza)

## Funzionalità

- Pulsante personalizzato basato su `QPushButton`
- Effetto glow animato tramite `QGraphicsDropShadowEffect`
- Transizione fluida del blur in entrata e in uscita (effetto bloom)
- Animazione del colore del glow al passaggio del mouse
- Facile integrazione in qualsiasi progetto PyQt5

## Requisiti

- Python 3.7+
- PyQt5

## Installazione e Avvio

Installa le dipendenze:

```bash
pip install PyQt5
```

Clona il repository ed esegui il progetto:

```bash
git clone https://github.com/GiovanniP9/BeFirstKC---GUI-for-Kaggle-Dataset.git
python main.py
```

## Struttura del Codice

- **widgets.py**: Contiene la classe `AnimatedButton` con le animazioni e l'effetto glow.
- **main.py**: Esempio di utilizzo in una finestra base PyQt5.

## Licenza

Distribuito con licenza MIT. Vedi il file `LICENSE` per maggiori dettagli.

# Fondamenti-di-AI
## Autori
- [Alessia](https://github.com/Alessiaaleale)
- [Emanuele](https://github.com/Leleart)
- [Giorgia](https://github.com/Giorgialopinto)

## Obiettivo

Il progetto ha come obiettivo la creazione di un modello di machine learning per classificare cellule tumorali in benigne o maligne. Le cellule tumorali possono essere benigne, caratterizzate da crescita limitata e localizzata, oppure maligne, con capacità aggressiva di proliferazione e metastasi. L'identificazione accurata di queste cellule è cruciale per diagnosi e trattamenti efficaci.

La classificazione si basa su caratteristiche morfologiche e biologiche, come forma e dimensione delle cellule, aspetto dei nuclei, grado di adesione cellulare e numero di mitosi.

Il modello proposto utilizza il classificatore k-NN (k-Nearest Neighbors) e diverse strategie di validazione (Holdout, Random Subsampling, Bootstrap) per garantire un'accurata valutazione. Il programma offre opzioni interattive per personalizzare il processo, come ad esempio la possibilità di scegliere una o più metriche di validazione.

## Panoramica Dettagliata del Dataset, version_1.csv:

- Numero di Campioni: 693 campioni.

- Numero di Caratteristiche: 13 caratteristiche per campione.

## Caratteristiche del Dataset:
1. **Blood Pressure:** pressione sanguigna registrata (dato aggiuntivo, non direttamente correlato alle cellule).
2. **Mitoses:** frequenza delle mitosi, indicativa del grado di proliferazione cellulare.
3. **Sample code number:** identificativo univoco per ogni campione.
4. **Normal Nucleoli:** numero di nucleoli normali presenti nelle cellule.
5. **Single Epithelial Cell Size:** dimensione della singola cellula epiteliale, indicatore di regolarità.
6. **Uniformity of Cell Size:** uniformità delle dimensioni cellulari, valore più alto può indicare malignità.
7. **Clump Thickness:** spessore del gruppo di cellule, utilizzato per valutare la densità dei campioni.
8. **Heart Rate:** frequenza cardiaca (dato aggiunto per scopi di studio, non strettamente legato alla classificazione).
9. **Marginal Adhesion:** capacità delle cellule di aderire tra loro.
10. **Bland Chromatin:** cromatina omogenea, legata all'aspetto dei nuclei delle cellule.
11. **classtype_v1:** classificazione delle cellule tumorali (2 per benigno, 4 per maligno).
12. **Uniformity of Cell Shape:** uniformità della forma delle cellule, importante per identificare alterazioni morfologiche.
13. **Bare Nucleix_wrong:** nuclei scoperti (probabile errore di digitazione per "Bare Nuclei").


## **Come eseguire il codice**
Eseguire i seguenti comandi nella directory principale del progetto.

_Clona il repository:_
```python
git clone https://github.com/Alessiaaleale/Fondamenti-di-AI
```
_Crea un ambiente virtuale:_
```python
python -m venv env
```
_Attiva l'ambiente virtuale:_
```python
.\env\Scripts\activate
```
Installa le dipendenze: indispensabile installare tutte le dipendenze. Questo comando installerà tutte le librerie necessarie, come numpy, pandas, matplotlib, e altre ancora.

```python
pip install -r requirements.txt
```

## **main.py**
Il file `main.py` è il nucleo del progetto. Si occupa di gestire l'intero processo, dalla lettura del dataset all'elaborazione dei dati, fino alla classificazione e alla visualizzazione dei risultati. Questo script è stato progettato per essere interattivo e offre numerose opzioni configurabili.

## **Input dell'Utente**
Il programma richiederà di inserire:
- Il nome del file con estensione.
- La colonna target.
- La colonna indice se esiste.
- La tipologia di split da utilizzare, di default _holdout_.
- Le percentuali di dati da usare nel test set.
- Il numero di interazioni se supportate dal metodo di split.
- Il parametro _k_ del _k-NN_.
- Quante e quali metriche da calcolare.

## **Caricamento del Dataset**:
Il programma è stato ideato per analizzare dataset in diversi formati di file. Per assicurare un corretto funzionamento, il dataset deve rispettare i seguenti requisiti:

### 1. Formati supportati
- Il file deve essere in uno dei seguenti formati:
  - `.csv`
  - `.xlsx`
  - `.json`
  - `.txt`
  - `.tsv`
- Se il file non rientra tra i formati supportati, verrà generato un errore.

### 2. Struttura del dataset
Oltre alle colonne precedentemente elencate, può contenere una colonna da utilizzare come indice del _pandas dataframe_. Questa colonna sarà utilizzata per identificare ogni campione univocamente.

### 3. Pulizia del dataset
- Settaggio della colonna specificata come indice.
- Eliminazione dei _NaN_ dalla colonna target.
- Fattorizzazione della colonna target.
- Eliminazione delle virgole con punti e conversione in float.
- Eliminazione delle colonne che contengono valori non numerici superiori all'80%.
- Restanti colonne con valori non numerici inferiori al 20% sostituiti con _NaN_.
- Rimpiazzo dei _NaN_ con media o mediana.
- Normalizzazione del dataframe.


## **Configurazione Interattiva**
Il programma permette di configurare diverse fasi del processo attraverso opzioni interattive:

### **1. Gestione dei Valori Mancanti**
L'utente può scegliere come trattare i valori mancanti nel dataset, selezionando una delle seguenti opzioni:
- `mean`: sostituisce i valori mancanti con la media delle colonne.
- `median`: sostituisce i valori mancanti con la mediana delle colonne.
Se non viene fornita una scelta valida, il programma utilizza automaticamente la strategia `mean`.

### **2. Validazione del Modello**
L'utente può scegliere una strategia di validazione per dividere i dati in set di training e test.
L'utente può selezionare una strategia di validazione per suddividere i dati in set di training e test:

1. **Holdout**: suddivide i dati in due parti, una per il training e una per il test. L'utente può specificare la percentuale di dati destinati al test.
2. **Random Subsampling**: esegue diverse divisioni casuali del dataset. L'utente può determinare il numero di iterazioni e la percentuale di dati per il test.
3. **Bootstrap**: genera più set di dati di training estraendo con sostituzione, ovvero uno stesso dato può essere selzionato più volte da un dataset. L'utente può specificare sia il numero di iterazioni che la percentuale.

## **Classificazione**
Il programma utilizza il classificatore **k-Nearest Neighbors (k-NN)** per distinguere tra tumori benigni e maligni.

## **Scelta di k per il Classificatore k-NN**
Durante l'esecuzione del programma, l'utente ha la possibilità di configurare il parametro _k_, che indica il numero di vicini da considerare per il classificatore _k-Nearest Neighbors (k-NN)_.

### **Dettagli sulla Configurazione**
- Il parametro _k_ stabilisce il numero di osservazioni più vicine che verranno considerate per effettuare la classificazione.

## **Metriche Calcolate**
Il progetto utilizza diverse metriche per valutare le prestazioni del modello di classificazione dei tumori. Le metriche da poter scegliere sono:
- **`Accuracy Rate`**: la percentuale di predizioni corrette rispetto al totale. Il suo valore ideale è vicino a 1.
    - Formula: `Accuracy Rate = TP + TN / (TP + TN + FP + FN)` dove:
     - `TP`: Veri positivi
     - `TN`: Veri negativi
     - `FP`: Falsi positivi
     - `FN`: Falsi negativi

- **`Error Rate`**: la percentuale di predizioni errate rispetto al totale. Il suo valore ideale è vicino allo 0.
    - Formula : `Error Rate = 1 - Accuracy Rate`

- **`Sensitivity`**: la capacità del modello di identificare correttamente i casi positivi . Il suo valore ideale è vicino a 1.
    - Formula: `Sensitivity = TP / (TP + FN)` dove:
     - `TP`: Veri positivi
     - `FN`: Falsi negativi

- **`Specificity`**: la capacità del modello di identificare correttamente i casi negativi (tumori benigni). Il suo valore ideale è vicino a 1.
    - Formula: `Specificity = TN / (TN + FP)` dove:
     - `TN`: Veri negativi
     - `FP`: Falsi positivi

- **`False Alarm Rate`**: la misura del tasso di falsi positivi, indica quante volte il modello classifica erroneamente una classe negativa come positiva.
    - Formula: `False Alarm Rate = FP / (FP + TN)` dove:
     - `TN`: Veri negativi
     - `FP`: Falsi positivi

- **`Miss Rate`**: la misura del tasso di falsi negativi, indica quante volte il modello non riesce a identificare correttamente una classe positiva.
    - Formula: `Miss Rate = FN / (FN + TP)` dove:
     - `FN`: Falsi negativi
     - `TP`: Veri positivi

- **`Geometric Mean`**: una misura dell'equilibrio tra Sensitivity e Specificity. Indica quanto il modello è bilanciato nell'identificazione delle due classi.
    - Formula: `Geometric Mean = √(Sensitivity × Specificity)`

- **`Area under the curve`**: la misura dell'area sottesa alla ROC Curve. Indica l'efficienza globale del modello.
    - Formula:$$ AUC = \sum_{i=1}^{n-1} \left( FPR_{i+1} - FPR_i \right) \left( \frac{TPR_i + TPR_{i+1}}{2} \right) $$
     - \( FPR \) è il False Positive Rate
     - \( TPR \) è il True Positive Rate
     - \( n \) è il numero totale dei punti sul ROC curve

Queste metriche offrono una valutazione esaustiva delle prestazioni del modello, considerando sia l'accuratezza complessiva sia la capacità di distinguere correttamente tra le due classi (positivi e negativi).

## **Visualizzazione e Salvataggio dei Risultati**
I risultati delle predizioni del modello saranno salvati:
- In un file Excel chiamato `metrics.xlsx`. Questo file conterrà le metriche di performance come Accuracy Rate, Sensitivity, Specificity, False alarm Rate, Miss Rate e Geometric Mean e la media di ogni metrica.
- In un _plot_ che mostra l'andamento delle metriche al crescere delle iterazioni.
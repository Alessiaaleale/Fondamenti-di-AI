# Fondamenti-di-AI
## Autori
- [Alessia](https://github.com/Alessiaaleale)
- [Emanuele](https://github.com/Leleart)
- [Giorgia](https://github.com/Giorgialopinto)

Come eseguire il codice 
============
Clona il repository
```python
git clone https://github.com/Alessiaaleale/Fondamenti-di-AI
```
Crea un ambiente virtuale
```python
python -m venv env
```
Attiva l'ambiente virtuale
```python
.\env\Scripts\activate
```
Installa le dipendenze
```python
pip install -r requirements.txt
```

## Obiettivo

Il progetto ha come obiettivo la creazione di un modello di machine learning per classificare cellule tumorali in benigne o maligne. Le cellule tumorali possono essere benigne, caratterizzate da crescita limitata e localizzata, oppure maligne, con capacità aggressiva di proliferazione e metastasi. L'identificazione accurata di queste cellule è cruciale per diagnosi e trattamenti efficaci.

La classificazione si basa su caratteristiche morfologiche e biologiche, come forma e dimensione delle cellule, aspetto dei nuclei, grado di adesione cellulare e numero di mitosi.

Il modello proposto utilizza il classificatore k-NN (k-Nearest Neighbors) e diverse strategie di validazione (Holdout, Random Subsampling, Bootstrap) per garantire un'accurata valutazione. Il programma offre opzioni interattive per personalizzare il processo, come ad esempio la possibilità di scegliere una o più metriche di validazione.
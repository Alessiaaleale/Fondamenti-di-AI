import numpy as np

class MetricsCalculator:
    """
    Classe per calcolare le metriche di classificazione.

    Attributi
    ----------
    confusion_matrix : list
        Lista contenente i valori della matrice di confusione [TP, TN, FP, FN].
    ypred : array
        Array contenente le predizioni del modello.
    ytest : array
        Array contenente i valori reali del set di test.
    """

    def __init__(self, confusion_matrix, ypred, ytest, target_column):
        """
        Inizializza un'istanza di MetricsCalculator con i dati di confusione, le predizioni e i valori di test.

        Parametri
        ----------
        confusion_matrix : list
            Lista contenente i valori della matrice di confusione [TP, TN, FP, FN].
        ypred : array
            Array contenente le predizioni del modello.
        ytest : array
            Array contenente i valori reali del set di test.

        """
        self.confusion_matrix = confusion_matrix
        self.tp, self.tn, self.fp, self.fn = confusion_matrix
        self.ypred = ypred
        self.ytest = ytest[target_column].tolist()
        for i in [self.tp, self.tn, self.fp, self.fn]:
            if i < 0:
                raise ValueError("I valori della matrice di confusione non possono essere negativi.")
        if self.tp and self.tn and self.fp and self.fn == 0:
            raise ValueError("La matrice di confusione non può contenere solo zeri.")

    def accuracy_rate(self) -> float:
        """
        Calcola il tasso di accuratezza.

        Return
        -------
        float
            Il tasso di accuratezza.
        """
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def error_rate(self) -> float:
        """
        Calcola il tasso di errore.

        Return
        -------
        float
            Il tasso di errore.
        """
        return 1 - self.accuracy_rate()

    def sensitivity(self) -> float:
        """
        Calcola la sensibilità (tasso di veri positivi).

        Return
        -------
        float
            Il valore della sensibilità.
        """
        if (self.tp + self.fn) > 0:
            return self.tp / (self.tp + self.fn)
        else:
            raise ValueError("Divisione per zero")

    def specificity(self) -> float:
        """
        Calcola la specificità (tasso di veri negativi).

        Return
        -------
        float
            Il valore della specificità.
        """
        if (self.tn + self.fp) > 0:
            return self.tn / (self.tn + self.fp)
        else:
            raise ValueError("Divisione per zero")

    def false_alarm_rate(self) -> float:
        """
        Calcola il tasso di falsi allarmi.

        Return
        -------
        float
            Il tasso di falsi allarmi.
        """
        if (self.fp + self.tn) > 0:
            return self.fp / (self.fp + self.tn)
        else:
            raise ValueError("Divisione per zero")

    def miss_rate(self) -> float:
        """
        Calcola il tasso di mancate rilevazioni.

        Return
        -------
        float
            Il tasso di mancate rilevazioni.
        """
        if (self.fn + self.tp) > 0:
            return self.fn / (self.fn + self.tp)
        else:
            raise ValueError("Divisione per zero")

    def geometric_mean(self) -> float:
        """
        Calcola la media geometrica.

        Return
        -------
        float
            Il valore della media geometrica.
        """
        sens = self.sensitivity()
        spec = self.specificity()
        return np.sqrt(sens * spec)

    def auc(self) -> float:
        """
        Calcola l'area sotto la curva (AUC) usando TP, TN, FP, FN.

        Return
        -------
        float
            Il valore dell'area sotto la curva (AUC).
        """
        fpr, tpr = [], []
        thresholds = np.linspace(0, 1, 10)

        for m in thresholds:
            tp = sum(1 for y, pred in zip(self.ytest, self.ypred) if y == 1.0 and pred >= m)
            tn = sum(1 for y, pred in zip(self.ytest, self.ypred) if y == 0.0 and pred < m)
            fp = sum(1 for y, pred in zip(self.ytest, self.ypred) if y == 0.0 and pred >= m)
            fn = sum(1 for y, pred in zip(self.ytest, self.ypred) if y == 1.0 and pred < m)
            tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr.append(tpr_value)
            fpr.append(fpr_value)

        # Ordinare i valori di FPR e TPR in base ai valori di FPR
        sorted_indices = np.argsort(fpr)
        fpr = np.array(fpr)[sorted_indices]
        fpr[0] = 0
        tpr = np.array(tpr)[sorted_indices]
        tpr[0] = 0
        # Calcolare l'area sotto la curva (AUC)
        auc_value = np.trapz(tpr, fpr)
        return auc_value

    def calculate_metrics(self, metrics) -> dict:
        """
        Calcola le metriche richieste.

        Parametri
        ----------
        metrics : list of str
            Lista delle metriche richieste da calcolare.

        Return
        -------
        dict
            Dizionario con i valori delle metriche calcolate.
        """
        if len(self.confusion_matrix) != 4:
            raise ValueError("La matrice di confusione deve contenere esattamente 4 valori: [TP, TN, FP, FN]")

        all_metrics = {
            'Accuracy Rate': self.accuracy_rate(),
            'Error Rate': self.error_rate(),
            'Sensitivity': self.sensitivity(),
            'Specificity': self.specificity(),
            'False Alarm Rate': self.false_alarm_rate(),
            'Miss Rate': self.miss_rate(),
            'Geometric Mean': self.geometric_mean(),
            'Area Under the Curve': self.auc()
        }
        if metrics:
            invalid_metrics = [metric for metric in metrics if metric not in all_metrics]
            if invalid_metrics:
                raise ValueError(f"Metriche non valide: {', '.join(invalid_metrics)}. Le opzioni disponibili sono: {', '.join(all_metrics.keys())}")
            return {metric: all_metrics[metric] for metric in metrics}

        return all_metrics
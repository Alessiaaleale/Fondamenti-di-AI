import numpy as np

class MetricsCalculator:
    """
    Classe per calcolare le metriche di classificazione.
    """
    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
        self.tp, self.tn, self.fp, self.fn = confusion_matrix
        for i in [self.tp, self.tn, self.fp, self.fn]:
            if i < 0:
                raise ValueError("I valori della matrice di confusione non possono essere negativi.")
        if self.tp and self.tn and self.fp and self.fn == 0:
            raise ValueError("La matrice di confusione non può contenere solo zeri.")

    def accuracy_rate(self) -> float:
        """
        Calcola l'accuracy rate.
        """
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def error_rate(self) -> float:
        """
        Calcola l'error rate.
        """
        return 1 - self.accuracy_rate()

    def sensitivity(self) -> float:
        """
        Calcola la sensitivity.
        """
        if (self.tp + self.fn) > 0:
            return self.tp / (self.tp + self.fn)
        else:
            raise ValueError("Divisione per zero")

    def specificity(self) -> float:
        """
        Calcola la specificity.
        """
        if (self.tn + self.fp) > 0:
            return self.tn / (self.tn + self.fp)
        else:
            raise ValueError("Divisione per zero")

    def false_alarm_rate(self) -> float:
        """
        Calcola la false alarm rate.
        """
        if (self.fp + self.tn) > 0:
            return self.fp / (self.fp + self.tn)
        else:
            raise ValueError("Divisione per zero")

    def miss_rate(self) -> float:
        """
        Calcola la miss rate.
        """
        if (self.fn + self.tp) > 0:
            return self.fn / (self.fn + self.tp)
        else:
            raise ValueError("Divisione per zero")

    def geometric_mean(self) -> float:
        """
        Calcola la geometric mean.
        """
        sens = self.sensitivity()
        spec = self.specificity()
        return np.sqrt(sens * spec)

    def auc(self) -> float:
        """
        Calcola l'area under the curve.
        """
        sens = self.sensitivity()
        spec = self.specificity()
        return (sens + spec) / 2

    def calculate_metrics(self, metrics=None)-> dict:
        """
        Calcola le metriche richieste.
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
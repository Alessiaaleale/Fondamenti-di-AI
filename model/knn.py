import numpy as np
import pandas as pd

class KNNClassifier:
    """
    Classe per implementare l'algoritmo K-Nearest Neighbors (KNN).
    """
    def __init__(self, k: int):
        """
        Inizializza la classe KNNClassifier con il numero di vicini k.

        Parametri:
        ----------
        k : int
            Numero di vicini da considerare per la classificazione.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Il valore di k deve essere un intero positivo.")
        self.k = k

    @staticmethod
    def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list:
        """
        Calcola la matrice di confusione.

        Parametri:
        ----------
        y_true : np.ndarray
            Array dei valori veri delle etichette.
        y_pred : np.ndarray
            Array dei valori predetti delle etichette.

        return:
        --------
        list:
            Lista contenente i valori [tn, tp, fn, fp] della matrice di confusione.
        """
        tp = tn = fp = fn = 0

        y_true = y_true.values
        y_pred = y_pred

        y_true_int = [int(value) for value in y_true]
        y_pred_int = [int(value) for value in y_pred]

        for true, pred in zip(y_true_int, y_pred_int):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 0:
                tn += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 1 and pred == 0:
                fn += 1
        return [tn, tp, fn, fp]

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calcola la distanza euclidea tra due punti.

        Parametri:
        ----------
        x1 : np.ndarray
            Primo punto.
        x2 : np.ndarray
            Secondo punto.

        return:
        --------
        float:
            La distanza euclidea tra i due punti.
        """
        return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))

    def knn(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame) -> list:
        """
        Predice la classe per un set di test.

        Parametri:
        ----------
        x_train : pd.DataFrame
            Dataset delle caratteristiche per il training.
        y_train : pd.DataFrame
            Etichette per il training.
        x_test : pd.DataFrame
            Dataset delle caratteristiche per il test.

        return:
        --------
        list:
            Lista delle classi predette per il set di test.
        """
        predictions = []
        for test_point in x_test.values:
            distances = []
            for i, train_point in enumerate(x_train.values):
                dist = self.euclidean_distance(test_point, train_point)
                distances.append((dist, y_train.iloc[i]))
            # Ordina per distanza e seleziona i k vicini più vicini
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            class_counts = {}
            for _, label in k_nearest:
                label = label.item()
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
            # Trova la classe con il massimo conteggio
            most_common = max(class_counts, key=class_counts.get)
            predictions.append(most_common)
        return predictions
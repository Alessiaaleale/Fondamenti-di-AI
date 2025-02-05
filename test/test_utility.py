import unittest
import numpy as np
from model.utility import classification_evaluation
import pandas as pd

class TestClassificationEvaluation(unittest.TestCase):
    def setUp(self):
        """Imposta dati di test validi."""
        self.k = 3
        np.random.seed(42)

        # Generazione casuale dei dati di test e train
        self.splits = [
            (
                pd.DataFrame(np.random.randint(0, 2, (10, 2))),  # xtrain come DataFrame
                pd.Series(np.random.randint(0, 2, 10)),         # ytrain come Series
                pd.DataFrame(np.random.randint(0, 2, (5, 2))),   # xtest come DataFrame
                pd.Series(np.random.randint(0, 2, 5))            # ytest come Series
            )
            for _ in range(3)
        ]

        # Cambiamo i nomi delle metriche con quelli accettati dalla funzione
        self.user_choice = ['Accuracy Rate', 'Sensitivity', 'Specificity']

    def test_knn_metrics_multiple_metrics(self):
        """Testa se knn_metrics restituisce più metriche correttamente."""
        result = classification_evaluation.knn_metrics(self.k, self.splits, self.user_choice)
        self.assertIn('Accuracy Rate', result)
        self.assertIn('Sensitivity', result)
        self.assertIn('Specificity', result)
        self.assertGreaterEqual(result['Accuracy Rate'], 0)
        self.assertLessEqual(result['Accuracy Rate'], 1)
        self.assertGreaterEqual(result['Sensitivity'], 0)
        self.assertLessEqual(result['Sensitivity'], 1)
        self.assertGreaterEqual(result['Specificity'], 0)
        self.assertLessEqual(result['Specificity'], 1)

    def test_knn_metrics_output_length(self):
        """Verifica che il numero di valori in output sia uguale al numero di metriche richieste."""
        result = classification_evaluation.knn_metrics(self.k, self.splits, self.user_choice)
        self.assertEqual(len(result), len(self.user_choice))  # Confronta le lunghezze

    def test_knn_metrics_returns_accuracy(self):
        """Verifica che, se si richiede solo 'Accuracy Rate', il dizionario contenga solo quella metrica."""
        result = classification_evaluation.knn_metrics(self.k, self.splits, ['Accuracy Rate'])
        self.assertEqual(list(result.keys()), ['Accuracy Rate'])  # Controlla che l'unica chiave sia 'Accuracy Rate'
        self.assertGreaterEqual(result['Accuracy Rate'], 0)
        self.assertLessEqual(result['Accuracy Rate'], 1)

    def test_knn_metrics_returns_mean_value(self):
        """Verifica che il valore restituito sia la media della metrica sui vari split."""
        user_choice = ['Accuracy Rate']  # Testiamo su Accuracy, ma vale per tutte
        result = classification_evaluation.knn_metrics(self.k, self.splits, user_choice)

        # Calcoliamo manualmente la media della metrica
        all_accuracies = []
        for xtrain, ytrain, xtest, ytest in self.splits:
            confusion_matrix = classification_evaluation.knn_metrics(self.k, [(xtrain, ytrain, xtest, ytest)], user_choice)
            all_accuracies.append(confusion_matrix['Accuracy Rate'])

        expected_mean = np.mean(all_accuracies)  # Calcoliamo la media attesa

        self.assertAlmostEqual(result['Accuracy Rate'], expected_mean, places=5)  # Confrontiamo con la media attesa

    def test_knn_metrics_handles_empty_splits(self):
        empty_splits = []
        result = classification_evaluation.knn_metrics(self.k, empty_splits, self.user_choice)
        self.assertEqual(result, {})
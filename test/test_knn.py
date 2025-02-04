import unittest
import pandas as pd
from model.knn import KNNClassifier

class TestKNNClassifier(unittest.TestCase):

    def setUp(self):
        self.knn = KNNClassifier(k=3)
        self.x_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1, 2, 3, 4, 5]
        })
        self.y_train = pd.Series([0, 0, 1, 1, 1])
        self.x_test = pd.DataFrame({
            'feature1': [1.5, 3.5],
            'feature2': [1.5, 3.5]
        })
        self.y_true = pd.Series([0, 1])
        self.y_pred = pd.Series([0, 1])

    def test_euclidean_distance(self):
        # Test della funzione di calcolo della distanza euclidea tra due punti
        dist = self.knn.euclidean_distance([1, 1], [4, 5])
        self.assertAlmostEqual(dist, 5.0)

    def test_knn(self):
        # Test della funzione di predizione KNN sul set di test
        predictions = self.knn.knn(self.x_train, self.y_train, self.x_test)
        self.assertEqual(predictions, [0, 1])

    def test_calculate_confusion_matrix(self):
        # Test della funzione di calcolo della matrice di confusione
        confusion_matrix = self.knn.calculate_confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(confusion_matrix, [1, 1, 0, 0])
import unittest
import numpy as np
import pandas as pd
from strategy_split import *

class TestSplitStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Crea un dataset di esempio che verrà usato per tutti i test.
        """
        np.random.seed(42)  # Per risultati riproducibili
        cls.X = pd.DataFrame(np.random.rand(100, 5), columns=["A", "B", "C", "D", "E"])
        cls.Y = pd.DataFrame(np.random.randint(0, 2, size=(100, 1)), columns=["Label"])

    def test_holdout_split(self):
        """
        Test per la strategia Holdout.
        """
        strategy = HoldoutSplit(percentage=0.2)
        splitter = Splitter(strategy)
        splits = splitter.execute(self.X, self.Y)

        # Verifica che ci sia un solo split (Holdout deve restituire una sola suddivisione)
        self.assertEqual(len(splits), 1)

        # Controlla che la dimensione sia corretta
        X_train, Y_train, X_test, Y_test = splits[0]
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(Y_train), 80)
        self.assertEqual(len(Y_test), 20)

        # Verifica che non ci siano dati duplicati
        self.assertEqual(len(set(X_train.index) & set(X_test.index)), 0)

    def test_random_subsampling_split(self):
        """
        Test per la strategia Random Subsampling.
        """
        strategy = RandomSubsamplingSplit(percentage=0.3, iterations=10)
        splitter = Splitter(strategy)
        splits = splitter.execute(self.X, self.Y)

        # Verifica che ci siano 10 iterazioni
        self.assertEqual(len(splits), 10)

        for X_train, Y_train, X_test, Y_test in splits:
            self.assertEqual(len(X_train), 70)
            self.assertEqual(len(X_test), 30)
            self.assertEqual(len(Y_train), 70)
            self.assertEqual(len(Y_test), 30)

    def test_bootstrap_split(self):
     """
     Test per la strategia Bootstrap.
     """
     strategy = BootstrapSplit(train_size=0.7, iterations=5)
     splitter = Splitter(strategy)
     splits = splitter.execute(self.X, self.Y)

     # Verifica che ci siano 5 iterazioni
     self.assertEqual(len(splits), 5)

     for X_train, Y_train, X_test, Y_test in splits:
         # Il training set dovrebbe avere la dimensione prevista (con ripetizioni)
         self.assertEqual(len(X_train), 70)

         # Il test set ha solo gli elementi che non sono stati campionati nel train
         unique_train_indices = set(X_train.index)
         unique_test_indices = set(X_test.index)

         # Non ci devono essere sovrapposizioni tra train e test
         self.assertTrue(unique_train_indices.isdisjoint(unique_test_indices))

         # Il numero di elementi nel test set è la differenza tra dataset originale e quelli selezionati per il train
         self.assertEqual(len(unique_test_indices), len(self.X) - len(unique_train_indices))


    def test_invalid_split_percentages(self):
        """
        Testa se le percentuali fuori dai limiti vengono gestite correttamente.
        """
        with self.assertRaises(ValueError):
            HoldoutSplit(percentage=1.5)

        with self.assertRaises(ValueError):
            RandomSubsamplingSplit(percentage=-0.2, iterations=5)

        with self.assertRaises(ValueError):
            BootstrapSplit(train_size=1.2, iterations=3)

    def test_invalid_iterations(self):
        """
        Testa se il numero di iterazioni non valido viene gestito correttamente.
        """
        with self.assertRaises(ValueError):
            RandomSubsamplingSplit(percentage=0.2, iterations=0)

        with self.assertRaises(ValueError):
            BootstrapSplit(train_size=0.8, iterations=-3)


if __name__ == "__main__":
    unittest.main()

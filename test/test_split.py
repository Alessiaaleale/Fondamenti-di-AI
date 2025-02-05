import unittest
import numpy as np
import pandas as pd
from evaluation.split import *

class TestSplit(unittest.TestCase):

    def setUp(self):
        """Crea un dataset di test."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=[f'Feature_{i}' for i in range(5)])
        self.Y = pd.DataFrame(np.random.randint(0, 2, size=(100, 1)), columns=['Label'])
        self.splitter = Split(percentage=0.2, iterations=3)

    def test_holdout(self):
        """Verifica che holdout restituisca uno split con la giusta dimensione."""
        splits = self.splitter.holdout(self.X, self.Y)
        X_train, Y_train, X_test, Y_test = splits[0]

        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(Y_test), 20)
        self.assertEqual(len(Y_train), 80)
        self.assertFalse(set(X_train.index) & set(X_test.index))  # Nessuna sovrapposizione

    def test_random_subsampling(self):
        """Verifica che random subsampling esegua il giusto numero di iterazioni."""
        splits = self.splitter.random_subsampling(self.X, self.Y)
        self.assertEqual(len(splits), 3)

        for X_train, Y_train, X_test, Y_test in splits:
            self.assertEqual(len(X_test), 20)
            self.assertEqual(len(X_train), 80)
            self.assertEqual(len(Y_test), 20)
            self.assertEqual(len(Y_train), 80)

    def test_bootstrap(self):
        """Verifica che bootstrap generi campioni con ripetizione e test set con gli esclusi."""
        splits = self.splitter.bootstrap(self.X, self.Y)
        self.assertEqual(len(splits), 3)

        for X_train, Y_train, X_test, Y_test in splits:
            self.assertEqual(len(X_train), 20)  # Training set con ripetizione
            self.assertTrue(len(X_test) > 0)  # Il test set non è vuoto
            self.assertEqual(len(Y_train), 20)
            self.assertEqual(len(Y_test), len(X_test))
            self.assertTrue(set(X_train.index).issubset(set(self.X.index)))  # Indici validi
            self.assertTrue(set(X_test.index).issubset(set(self.X.index)))


    def test_holdout_different_percentage(self):
        """Verifica che holdout funzioni con percentuali diverse."""
        for perc in [0.1, 0.3, 0.5]:
            splitter = Split(percentage=perc)
            splits = splitter.holdout(self.X, self.Y)
            X_train, Y_train, X_test, Y_test = splits[0]

            expected_test_size = int(len(self.X) * perc)
            self.assertEqual(len(X_test), expected_test_size)
            self.assertEqual(len(X_train), len(self.X) - expected_test_size)

    def test_random_subsampling_different_iterations(self):
        """Verifica che random subsampling funzioni con diverse iterazioni."""
        for iters in [1, 5, 10]:
            splitter = Split(percentage=0.2, iterations=iters)
            splits = splitter.random_subsampling(self.X, self.Y)
            self.assertEqual(len(splits), iters)

    def test_bootstrap_sample_size(self):
        """Verifica che il training set di bootstrap abbia la giusta dimensione."""
        for perc in [0.1, 0.25, 0.5]:
            splitter = Split(percentage=perc, iterations=3)
            splits = splitter.bootstrap(self.X, self.Y)
            for X_train, Y_train, X_test, Y_test in splits:
                self.assertEqual(len(X_train), int(len(self.X) * perc))
                self.assertTrue(len(X_test) > 0)
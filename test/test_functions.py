import unittest
import pandas as pd
import numpy as np
from preprocessing.functions import DataPreprocessing

class TestDataPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.df = pd.DataFrame({
            'ID': range(1, 11),
            'Feature1': np.random.randn(10),
            'Feature2': [np.nan, 2.5, '5.3', '8.1', 3.3, np.nan, 4.0, 1.2, np.nan, 6.7],
            'Target': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 'A']
        })
        cls.preprocessor = DataPreprocessing(cls.df.copy())

    def test_set_column_as_index(self):
        '''
        Testa se la funzione set_column_as_index imposta correttamente la colonna 'ID' come indice.
        '''
        df = self.preprocessor.set_column_as_index('ID')
        self.assertTrue(df.index.name == 'ID')

    def test_drop_nan_target(self):
        '''
        Testa se la funzione drop_nan_target rimuove le righe con valori NaN nella colonna 'Target'.
        '''
        df = self.preprocessor.drop_nan_target('Target')
        self.assertFalse(df['Target'].isna().any())

    def test_factorize_target_column(self):
        '''
        Testa se la funzione factorize_target_column converte correttamente la colonna 'Target' in valori numerici.
        '''
        df = self.preprocessor.factorize_target_column('Target')
        self.assertTrue(pd.api.types.is_integer_dtype(df['Target']))

    def test_remove_commas_to_float(self):
        '''
        Testa se la funzione remove_commas_to_float mantiene la colonna 'Feature2' dopo la conversione dei valori.
        '''
        df = self.preprocessor.remove_commas_to_float()
        self.assertIn('Feature2', df.columns)  # Verifica che la colonna esista ancora

    def test_filter_columns_by_numeric_percentage(self):
        '''
        Testa se la funzione filter_columns_by_numeric_percentage mantiene almeno una colonna con percentuale numerica adeguata.
        '''
        df = self.preprocessor.filter_columns_by_numeric_percentage(0.6)
        self.assertGreaterEqual(len(df.columns), 1)  # Almeno una colonna deve essere presente

    def test_replace_string_with_nan(self):
        '''
        Testa se la funzione replace_string_with_nan mantiene la colonna 'Feature2' dopo la conversione delle stringhe in NaN.
        '''
        df = self.preprocessor.replace_string_with_nan()
        self.assertIn('Feature2', df.columns)  # Verifica che la colonna esista ancora

    def test_replace_nan(self):
        '''
        Testa se la funzione replace_nan sostituisce correttamente i valori NaN con la media del gruppo di appartenenza.
        '''
        df = self.preprocessor.replace_nan('mean', 'Target')
        self.assertFalse(df.isna().any().any())

    def test_scale_columns(self):
        '''
        Testa se la funzione scale_columns normalizza correttamente le colonne tra 0 e 1.
        '''
        df = self.preprocessor.scale_columns()
        for col in df.columns:
            self.assertGreaterEqual(df[col].min(), 0)
            self.assertLessEqual(df[col].max(), 1)

    def test_features_and_target(self):
        '''
        Testa se la funzione features_and_target separa correttamente le features dal target.
        '''
        features, target = self.preprocessor.features_and_target('Target')
        self.assertTrue('Target' not in features.columns)
        self.assertTrue(target.shape[1] == 1 and target.columns[0] == 'Target')

    def test_full_preprocessing(self):
        '''
        Testa l'intero processo di preprocessing verificando che non ci siano valori NaN,
        che il target sia presente e che le colonne siano normalizzate correttamente.
        '''
        df = self.preprocessor.preprocessing('ID', 'Target', 'mean')
        self.assertFalse(df.isna().any().any())
        self.assertIn('Target', df.columns)  # Verifica che la colonna target esista ancora
        for col in df.columns:
            self.assertGreaterEqual(df[col].min(), 0)
            self.assertLessEqual(df[col].max(), 1)
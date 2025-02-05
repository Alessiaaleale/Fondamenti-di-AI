import unittest
from unittest.mock import patch
import pandas as pd
from preprocessing.data_parser import TSVFileOpener

class TestTSVFileOpener(unittest.TestCase):

    def setUp(self):
        self.file_opener = TSVFileOpener()

    @patch('pandas.read_csv')
    def test_open_file_success(self, mock_read_csv):
        # Simula un DataFrame che verrà restituito da pd.read_csv
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df

        # Percorso fittizio del file
        file_path = "test.tsv"

        # Chiamata al metodo open_file
        result = self.file_opener.open_file(file_path)

        # Verifica che il risultato sia il DataFrame simulato
        pd.testing.assert_frame_equal(result, mock_df)

        # Verifica che read_csv sia stato chiamato con il file giusto e con il separatore corretto
        mock_read_csv.assert_called_once_with(file_path, sep='\t')

    @patch('pandas.read_csv', side_effect=Exception("File not found"))
    def test_open_file_failure(self, mock_read_csv):
        # Percorso del file che non esiste
        file_path = "non_existent_file.tsv"

        # Chiamata al metodo open_file che dovrebbe fallire
        result = self.file_opener.open_file(file_path)

        # Verifica che il risultato sia None in caso di errore
        self.assertIsNone(result)

        # Verifica che read_csv sia stato chiamato con il file giusto e con il separatore corretto
        mock_read_csv.assert_called_once_with(file_path, sep='\t')
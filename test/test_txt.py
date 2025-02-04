import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from preprocessing.data_parser import *

class TestTextFileOpener(unittest.TestCase):

    def setUp(self):
        self.file_opener = TextFileOpener()

    @patch('builtins.open', new_callable=mock_open, read_data="col1,col2\n1,3\n2,4\n")
    @patch('pandas.read_csv')
    def test_open_file_success(self, mock_read_csv, mock_open_file):
        # Simula un DataFrame che verrà restituito da pd.read_csv
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df

        # Percorso fittizio del file
        file_path = "test.txt"

        # Chiamata al metodo open_file
        result = self.file_opener.open_file(file_path)

        # Verifica che il risultato sia il DataFrame simulato
        pd.testing.assert_frame_equal(result, mock_df)

        # Verifica che read_csv sia stato chiamato con i dati corretti
        mock_read_csv.assert_called_once()

    @patch('builtins.open', new_callable=mock_open, read_data="col1,col2\n1,3\n2,4\n")
    @patch('pandas.read_csv', side_effect=Exception("File format error"))
    def test_open_file_failure(self, mock_read_csv, mock_open_file):
        # Percorso del file che potrebbe avere problemi di formattazione
        file_path = "malformed_file.txt"

        # Chiamata al metodo open_file
        result = self.file_opener.open_file(file_path)

        # Verifica che il risultato sia None in caso di errore
        self.assertIsNone(result)

        # Verifica che read_csv sia stato chiamato con i dati corretti
        mock_read_csv.assert_called_once()
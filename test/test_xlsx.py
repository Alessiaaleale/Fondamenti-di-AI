import unittest
from unittest.mock import patch
import pandas as pd
from preprocessing.data_parser import *

class TestCSVFileOpener(unittest.TestCase):

    def setUp(self):
        self.file_opener = ExcelFileOpener()

    @patch('pandas.read_excel')
    def test_open_file_success(self, mock_read_excel):
        # Simula un DataFrame che verrà restituito da pd.read_excel
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_excel.return_value = mock_df

        # Percorso fittizio del file
        file_path = "test.xlsx"

        # Chiamata al metodo open_file
        result = self.file_opener.open_file(file_path)

        # Verifica che il risultato sia il DataFrame simulato
        pd.testing.assert_frame_equal(result, mock_df)
        mock_read_excel.assert_called_once_with(file_path)  # Verifica che read_excel sia stato chiamato con il file giusto

    @patch('pandas.read_excel', side_effect=Exception("File not found"))
    def test_open_file_failure(self, mock_read_excel):
        # Percorso del file che non esiste
        file_path = "non_existent_file.xlsx"

        # Chiamata al metodo open_file che dovrebbe fallire
        result = self.file_opener.open_file(file_path)

        # Verifica che il risultato sia None in caso di errore
        self.assertIsNone(result)
        mock_read_excel.assert_called_once_with(file_path)  # Verifica che read_excel sia stato chiamato con il file giusto
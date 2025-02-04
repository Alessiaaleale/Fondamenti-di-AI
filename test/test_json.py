import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from preprocessing.data_parser import JSONFileOpener

class TestJSONFileOpener(unittest.TestCase):

    def setUp(self):
        self.file_opener = JSONFileOpener()

    @patch('builtins.open', new_callable=mock_open, read_data='{"col1": [1, 2], "col2": [3, 4]}')
    @patch('json.load')
    def test_open_file_success(self, mock_json_load, mock_file):
        # Simula il ritorno di json.load
        mock_json_load.return_value = {"col1": [1, 2], "col2": [3, 4]}

        # Percorso fittizio del file
        file_path = "test.json"

        # Chiamata al metodo open_file
        result = self.file_opener.open_file(file_path)

        expected_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Verifica che il risultato sia il DataFrame atteso
        pd.testing.assert_frame_equal(result, expected_df)

        # Verifica che open e json.load siano stati chiamati correttamente
        mock_file.assert_called_once_with(file_path, 'r', encoding='utf-8')
        mock_json_load.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load', side_effect=Exception("File not found"))
    def test_open_file_failure(self, mock_json_load, mock_file):
        # Percorso del file che non esiste
        file_path = "non_existent_file.json"

        # Chiamata al metodo open_file che dovrebbe fallire
        result = self.file_opener.open_file(file_path)

        # Verifica che il risultato sia None in caso di errore
        self.assertIsNone(result)

        # Verifica che open sia stato chiamato
        mock_file.assert_called_once_with(file_path, 'r', encoding='utf-8')
        mock_json_load.assert_called_once()
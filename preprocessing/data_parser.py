from abc import ABC, abstractmethod
import pandas as pd
import json
import io
import os

class FileOpenerStrategy(ABC):
    '''
    Classe astratta che definisce l'interfaccia
    '''
    @abstractmethod
    def open_file(self, file_path: str) -> pd.DataFrame:
        pass


class CSVFileOpener(FileOpenerStrategy):
    '''
    Classe che implementa il parser per i file csv
    '''
    def open_file(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            print(f"[CSVFileOpener] Opened CSV file: {file_path}")
            return df
        except Exception as e:
            print(f"[CSVFileOpener] Failed to open CSV file: {file_path} - {e}")
            return None

class ExcelFileOpener(FileOpenerStrategy):
    '''
    Classe che implementa il parser per i file excel
    '''
    def open_file(self, file_path: str)-> pd.DataFrame:
        try:
            df = pd.read_excel(file_path)
            print(f"[ExcelFileOpener] Opened Excel file: {file_path}")
            return df
        except Exception as e:
            print(f"[ExcelFileOpener] Failed to open Excel file: {file_path} - {e}")
            return None

class TextFileOpener(FileOpenerStrategy):
    '''
    Classe che implementa il parser per i file di testo
    '''
    def open_file(self, file_path: str)-> pd.DataFrame:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()
            # Controlla se è un file di testo csv (comma o tab-separated)
            first_line = content[0]
            if ',' in first_line:
                delimiter = ','
            elif '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = ' '  # default spazio bianco se non viene trovato un delimitatore
            df = pd.read_csv(io.StringIO(''.join(content)), delimiter=delimiter)
            print(f"[TextFileOpener] Opened text file: {file_path}")
            return df
        except Exception as e:
            print(f"[TextFileOpener] Failed to open text file: {file_path} - {e}")
            return None

class TSVFileOpener(FileOpenerStrategy):
    '''
    Classe che implementa il parser per i file tsv
    '''
    def open_file(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, sep='\t')
            print(f"[TSVFileOpener] Opened TSV file: {file_path}")
            return df
        except Exception as e:
            print(f"[TSVFileOpener] Failed to open TSV file: {file_path} - {e}")
            return None

class JSONFileOpener(FileOpenerStrategy):
    '''
    Classe che implementa il parser per i file JSON
    '''
    def open_file(self, file_path: str) -> pd.DataFrame:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"[JSONFileOpener] Opened JSON file: {file_path}")
            df = pd.DataFrame(data)
            # Tentare di convertire le colonne a int64 ove possibile
            for column in df.columns:
                try:
                    df[column] = pd.to_numeric(df[column])
                    if df[column].dtype == 'float64':
                        df[column] = df[column].astype('int64')
                except (ValueError, TypeError):
                    pass  # Lascia la colonna invariata se non può essere convertita
            return df

        except Exception as e:
            print(f"[JSONFileOpener] Failed to open JSON file: {file_path} - {e}")
            return None


class FileOpener:
    '''
    Restituisce la strategia di apertura file appropriata in base all'estensione del file
    '''
    def get_file_opener(self, file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Il file {file_path} non è presente nella directory.")
        ext = file_path[file_path.rfind('.'):]
        if ext == '.csv':
            return CSVFileOpener()
        elif ext in ['.xlsx', '.xls']:
            return ExcelFileOpener()
        elif ext == '.txt':
            return TextFileOpener()
        elif ext == '.tsv':
            return TSVFileOpener()
        elif ext == '.json':
            return JSONFileOpener()
        else:
            raise ValueError(f"No strategy found for file: {file_path}")

    def open(self, file_path: str) -> pd.DataFrame:
        file_opener = self.get_file_opener(file_path)
        return file_opener.open_file(file_path)
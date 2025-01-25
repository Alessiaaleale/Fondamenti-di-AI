from abc import ABC, abstractmethod
import pandas as pd
import json
import io  

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
            df = pd.json_normalize(data)  # Converte JSON a Dataframe
            return df
        except Exception as e:
            print(f"[JSONFileOpener] Failed to open JSON file: {file_path} - {e}")
            return None


def open_file(file_path: str) -> pd.DataFrame:
    '''
    La funzione seleziona la strategia appropriata in base all'estensione del file
    '''
    file_path_to_strategy = {
        '.csv': CSVFileOpener(),
        '.xlsx': ExcelFileOpener(),
        '.xls': ExcelFileOpener(),
        '.txt': TextFileOpener(),
        '.tsv': TSVFileOpener(),
        '.json': JSONFileOpener(),  
    }
    ext = file_path[file_path.rfind('.'):] #Estrae l'estensione del file 
    strategy = file_path_to_strategy.get(ext, None)
    if strategy: #Se viene trovata una strategia, chiama il metodo open_file della strategia selezionata.
        return strategy.open_file(file_path)
    else:
        raise ValueError(f"No strategy found for file : {file_path}")
        
import pandas as pd

class DataPreprocessing:
    """
    Classe per il preprocessing dei dati.
    """
    def __init__(self, df) :
        self.df = df
    def drop_nan_target(self, col_name) -> pd.DataFrame:
        """
        Rimuove le righe con valori NaN nella colonna target.
        """
        self.df = self.df.dropna(subset=[col_name])
    def remove_commas_to_float(self) -> pd.DataFrame:
        """
        Sostituisce le virgole con i punti e converte le colonne in float.
        """
        self.df = self.df.replace(',', '.', regex=True).astype(float)
    def replace_nan(self, method_fill_nan, col_name) -> pd.DataFrame:
        """
        Sostituisce i valori NaN con la media o la mediana.
        """
        if method_fill_nan == 'mean':
            for column in self.df.loc[:, self.df.columns != col_name]:
                # Raggruppa il DataFrame in base ai valori della colonna col_name
                # Per ogni gruppo, seleziona la colonna specificata
                # e riempe i valori mancanti con la media dei valori presenti in quel gruppo
                self.df[column] = self.df.groupby(col_name)[column].transform(lambda x: x.fillna(x.mean()))
        elif method_fill_nan == 'median':
            for column in self.df.loc[:, self.df.columns != col_name]:
                # Raggruppa il DataFrame in base ai valori della colonna col_name
                # Per ogni gruppo, seleziona la colonna specificata
                # e riempe i valori mancanti con la mediana dei valori presenti in quel gruppo
                self.df[column] = self.df.groupby(col_name)[column].transform(lambda x: x.fillna(x.median()))
        else:
            raise ValueError("Method must be 'mean' or 'median'")
    def scale_columns(self)-> pd.DataFrame:
        """"
        Normalizza le colonne.
        """
        for column in self.df.columns:
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val- min_val)

    def preprocessing(self, col_name, method_fill_nan)-> pd.DataFrame:
        """
        Esegue il preprocessing dei dati.
        """
        df = self.drop_nan_target(col_name)
        df = self.remove_commas_to_float()
        df = self.replace_nan(method_fill_nan, col_name)
        df = self.scale_columns()
        df = self.df
        return df
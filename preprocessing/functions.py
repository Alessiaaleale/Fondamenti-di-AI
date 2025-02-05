import pandas as pd

class DataPreprocessing:
    """
    Classe per il preprocessing dei dati.
    """
    def __init__(self, df) :
        self.df = df

    def set_column_as_index(self, index_col) -> pd.DataFrame:
        """
        Imposta la colonna specificata come indice del Dataframe
        """
        if index_col in self.df.columns:
            self.df = self.df.set_index(index_col)
        else:
            print(f"La colonna '{index_col}' non è stata trovata. L'indice verrà impostato in automatico.")
        return self.df
    
    def drop_nan_target(self, target_column) -> pd.DataFrame:
        """
        Rimuove le righe con valori NaN nella colonna target.
        """
        self.df = self.df.dropna(subset=[target_column])
        return self.df
    
    def factorize_target_column(self, target_column) -> pd.DataFrame:
        """
        Sostituisce i valori della colonna target con valori numerici.
        """
        self.df[target_column] = pd.factorize(self.df[target_column])[0]
        return self.df
    
    def remove_commas_to_float(self) -> pd.DataFrame:
        """
        Sostituisce le virgole con i punti e converte le colonne in float.
        """
        try:
            self.df = self.df.replace(',', '.', regex=True).astype(float)
        except ValueError:
            pass
        return self.df
    
    def filter_columns_by_numeric_percentage(self, threshold: float = 0.8) -> pd.DataFrame:
        """
        Mantiene solo le colonne con una percentuale di valori numerici maggiore di 'threshold'.
        """
        # Calcolare la percentuale di valori numerici per ogni colonna
        numeric_counts = self.df.map(lambda x: isinstance(x, (int, float))).sum()
        total_counts = len(self.df)
        numeric_percentage = numeric_counts / total_counts
        # Filtrare le colonne in base alla percentuale di valori numerici
        self.df = self.df.loc[:, numeric_percentage >= threshold]
        return self.df
    
    def replace_string_with_nan(self) -> pd.DataFrame:
        """
        Sostituisce le stringhe con valori NaN.
        """
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        return self.df
    
    def replace_nan(self, method_fill_nan, target_column) -> pd.DataFrame:
        """
        Sostituisce i valori NaN con la media o la mediana.
        """
        if method_fill_nan == 'mean':
            for column in self.df.loc[:, self.df.columns != target_column]:
                # Raggruppa il DataFrame in base ai valori della colonna target_column
                # Per ogni gruppo, seleziona la colonna specificata
                # e riempe i valori mancanti con la media dei valori presenti in quel gruppo
                self.df[column] = self.df.groupby(target_column)[column].transform(lambda x: x.fillna(x.mean()))
        elif method_fill_nan == 'median':
            for column in self.df.loc[:, self.df.columns != target_column]:
                # Raggruppa il DataFrame in base ai valori della colonna target_column
                # Per ogni gruppo, seleziona la colonna specificata
                # e riempe i valori mancanti con la mediana dei valori presenti in quel gruppo
                self.df[column] = self.df.groupby(target_column)[column].transform(lambda x: x.fillna(x.median()))
        else:
            method_fill_nan = 'mean'
            print("Metodo non valido. Utilizzata 'mean' di default.")
        return self.df
    
    def scale_columns(self)-> pd.DataFrame:
        """"
        Normalizza le colonne.
        """
        for column in self.df.columns:
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val- min_val)
        return self.df

    def features_and_target(self, target_column) -> tuple:
        """
        Separa le features e il target in un DataFrame.
        """
        features = self.df.iloc[:,self.df.columns != target_column]
        target = self.df.iloc[:,self.df.columns == target_column]
        return features, target
    
    def preprocessing(self, index_col, target_column, method_fill_nan)-> pd.DataFrame:
        """
        Esegue il preprocessing dei dati.
        """
        if target_column in self.df.columns:
            pass
        else:
            raise ValueError("La colonna non è presente del dataset.")
        df = self.set_column_as_index(index_col)
        df = self.drop_nan_target(target_column)
        df = self.factorize_target_column(target_column)
        df = self.remove_commas_to_float()
        df = self.filter_columns_by_numeric_percentage()
        df = self.replace_string_with_nan()
        df = self.replace_nan(method_fill_nan, target_column)
        df = self.scale_columns()
        df = self.df
        return df
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class SplitStrategy(ABC):
    """
    Classe astratta per definire l'interfaccia delle strategie di suddivisione dei dati.
    """
    @abstractmethod
    def split(self, X: pd.DataFrame, Y: pd.DataFrame) -> list:
        """
        Metodo astratto per suddividere i dati in training e test.

        Parametri:
        ----------
        X : pd.DataFrame
            DataFrame delle caratteristiche.
        Y : pd.DataFrame
            DataFrame delle etichette.

        Return:
        --------
        list:
            Lista di tuple (X_train, Y_train, X_test, Y_test).
        """
        pass

class HoldoutSplit(SplitStrategy):
    """
    Classe che implementa la strategia Holdout per suddividere i dati in training e test.
    """
    def __init__(self, percentage: float = 0.25):
        """
        Inizializza la classe HoldoutSplit con la percentuale del dataset da utilizzare come test set.

        Parametri:
        ----------
        percentage : float, optional
            Percentuale del dataset da utilizzare come test set (default è 0.25).
        """
        if not (0 < percentage < 1):
            raise ValueError("La percentuale deve essere un valore tra 0 e 1 (esclusi).")
        self.percentage = percentage

    def split(self, X: pd.DataFrame, Y: pd.DataFrame) -> list:
        """
        Suddivide i dati in training e test set utilizzando Holdout.

        Parametri:
        ----------
        X : pd.DataFrame
            DataFrame delle caratteristiche.
        Y : pd.DataFrame
            DataFrame delle etichette.

        Return:
        --------
        list:
            Lista di una tupla (X_train, Y_train, X_test, Y_test).
        """
        samples = len(X)
        test_count = int(self.percentage * samples)
        indices = np.arange(samples)
        np.random.shuffle(indices)

        train_indices = indices[:-test_count]
        test_indices = indices[-test_count:]

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]

        return [(X_train, Y_train, X_test, Y_test)]

class RandomSubsamplingSplit(SplitStrategy):
    """
    Classe che implementa la strategia Random Subsampling per suddividere i dati in più iterazioni.
    """
    def __init__(self, percentage: float = 0.25, iterations: int = 5):
        """
        Inizializza la classe RandomSubsamplingSplit con la percentuale del dataset da utilizzare come test set
        e il numero di iterazioni.

        Parametri:
        ----------
        percentage : float, optional
            Percentuale del dataset da utilizzare come test set (default è 0.25).
        iterations : int, optional
            Numero di iterazioni per il random subsampling (default è 5).
        """
        if not (0 < percentage < 1):
            raise ValueError("La percentuale deve essere un valore tra 0 e 1 (esclusi).")
        if iterations < 1:
            raise ValueError("Le iterazioni devono essere un intero positivo.")
        self.percentage = percentage
        self.iterations = iterations

    def split(self, X: pd.DataFrame, Y: pd.DataFrame) -> list:
        """
        Esegue il random subsampling suddividendo i dati in più iterazioni.

        Parametri:
        ----------
        X : pd.DataFrame
            DataFrame delle caratteristiche.
        Y : pd.DataFrame
            DataFrame delle etichette.

        Return:
        --------
        list:
            Lista di tuple (X_train, Y_train, X_test, Y_test) per ogni iterazione.
        """
        samples = len(X)
        test_count = int(self.percentage * samples)
        splits = []

        for _ in range(self.iterations):
            indices = np.arange(samples)
            np.random.shuffle(indices)

            train_indices = indices[:-test_count]
            test_indices = indices[-test_count:]

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]

            splits.append((X_train, Y_train, X_test, Y_test))

        return splits

class BootstrapSplit(SplitStrategy):
    """
    Classe che implementa la strategia Bootstrap per suddividere i dati con campionamento con ripetizione.
    """
    def __init__(self, train_size: float = 0.75, iterations: int = 5):
        """
        Inizializza la classe BootstrapSplit con la dimensione del training set e il numero di iterazioni.

        Parametri:
        ----------
        train_size : float, optional
            Dimensione del training set come percentuale del dataset (default è 0.75).
        iterations : int, optional
            Numero di iterazioni per il bootstrap (default è 5).
        """
        if not (0 < train_size <= 1):
            raise ValueError("La dimensione del training set deve essere un valore tra 0 e 1 (incluso).")
        if iterations < 1:
            raise ValueError("Le iterazioni devono essere un intero positivo.")
        self.train_size = train_size
        self.iterations = iterations

    def split(self, X: pd.DataFrame, Y: pd.DataFrame) -> list:
        """
        Esegue il bootstrap suddividendo i dati con campionamento con ripetizione.

        Parametri:
        ----------
        X : pd.DataFrame
            DataFrame delle caratteristiche.
        Y : pd.DataFrame
            DataFrame delle etichette.

        Return:
        --------
        list:
            Lista di tuple (X_train, Y_train, X_test, Y_test) per ogni iterazione.
        """
        samples = len(X)
        train_count = int(self.train_size * samples)
        splits = []

        for _ in range(self.iterations):
            train_indices = np.random.choice(samples, size=train_count, replace=True)
            test_indices = np.setdiff1d(np.arange(samples), train_indices)

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]

            splits.append((X_train, Y_train, X_test, Y_test))

        return splits

class Splitter:
    """
    Classe per gestire la suddivisione dei dati utilizzando diverse strategie di splitting.
    """
    def __init__(self, strategy: SplitStrategy):
        """
        Inizializza il contesto con una strategia specifica.

        Parametri:
        ----------
        strategy : SplitStrategy
            Oggetto di una classe che implementa SplitStrategy.
        """
        self.strategy = strategy

    def execute(self, X: pd.DataFrame, Y: pd.DataFrame) -> list:
        """
        Esegue lo split utilizzando la strategia scelta.

        Parametri:
        ----------
        X : pd.DataFrame
            DataFrame delle caratteristiche.
        Y : pd.DataFrame
            DataFrame delle etichette.

        Return:
        --------
        list:
            Lista di tuple (X_train, Y_train, X_test, Y_test) per ogni iterazione.
        """
        return self.strategy.split(X, Y)
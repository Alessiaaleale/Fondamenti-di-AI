import numpy as np

class Split:
    def __init__(self, percentage=0.25, iterations=5):
        """
        percentage : Proporzione del dataset da utilizzare come test set (per holdout e random subsampling)
        iterations: Numero di iterazioni per random subsampling e bootstrap
        """
        self.percentage = percentage
        self.iterations = iterations

    def holdout(self, X, Y) -> list:
        """
        Il metodo esegue holdout per creare split di training e test.
        ---Parametri---
        X: Dataset delle caratteristiche
        Y: Etichette
        return: Lista di una tupla (x_train, y_train, x_test, y_test)
        """
        splits = []
        samples = len(X)
        percentage = int(self.percentage * samples)  # Numero di campioni da dedicare al test set
        indices = np.arange(samples)  # Indice le righe da 0 a samples-1
        np.random.shuffle(indices)  # Mescola casualmente l'array di indici

        train_indices = indices[:-percentage]
        test_indices = indices[-percentage:]

        X_train = X.iloc[train_indices, :]
        X_test = X.iloc[test_indices, :]
        Y_train = Y.iloc[train_indices, :]
        Y_test = Y.iloc[test_indices, :]

        splits.append((X_train, Y_train, X_test, Y_test))
        return splits

    def random_subsampling(self, X, Y) -> list:
        """
        Il metodo esegue random subsampling per n iterazioni.
        ---Parametri---
        X: Dataset delle caratteristiche
        Y: Etichette
        return: Lista di tuple (x_train, y_train, x_test, y_test) per ogni iterazione.
        """
        samples = len(X)
        percentage  = int(self.percentage * samples) #numero di campioni da dedicare al test set
        splits = []

        for i in range(self.iterations):
            indices = np.arange(samples)
            np.random.shuffle(indices)

            train_indices = indices[:-percentage]
            test_indices = indices[-percentage:]

            X_train = X.iloc[train_indices]
            Y_train = Y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            Y_test = Y.iloc[test_indices]

            splits.append((X_train, Y_train, X_test, Y_test))
        return splits

    def bootstrap(self, X, Y) -> list:
        """
        Esegue il bootstrap per creare split di training e test.
        :param X: DataFrame delle caratteristiche.
        :param Y:  Etichette.
        :return: Lista di tuple (x_train, y_train, x_test, y_test) per ogni iterazione.
        """
        n_train = int(len(X) * self.percentage)
        campioni = len(X)
        splits = []

        for i in range(self.iterations):
            # Campionamento con ripetizione per ottenere il training set
            indici_train = np.random.choice(campioni, size=n_train, replace=True)
            indici_test = np.setdiff1d(np.arange(campioni), indici_train)  # Elementi non selezionati per il test set

            X_train = X.iloc[indici_train]
            Y_train = Y.iloc[indici_train]
            X_test = X.iloc[indici_test]
            Y_test = Y.iloc[indici_test]

            splits.append((X_train, Y_train, X_test, Y_test))
        return splits
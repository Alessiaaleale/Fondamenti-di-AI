from evaluation.split import Split
import pandas as pd

class InputManager:
    """
    Classe per gestire le interazioni con l'utente e raccogliere le sue scelte.
    """
    @staticmethod
    def get_user_choice_split(X: pd.DataFrame, Y: pd.DataFrame) -> list:
        """
        Chiede all'utente di scegliere il tipo di split da utilizzare.

        Parametri:
        ----------
        X : pd.DataFrame
            DataFrame delle caratteristiche.
        Y : pd.DataFrame
            DataFrame delle etichette.

        return:
        --------
        list:
            Lista di tuple (X_train, Y_train, X_test, Y_test) in base alla scelta dell'utente.
        """
        print("Scegli tipologia di split del dataset vuoi utilizzare")
        print("1. Holdout")
        print("2. Random subsampling")
        print("3. Bootstrap")
        choice = input("Inserisci il numero della tua scelta: ")

        # Input per la percentuale di dati nel set di test
        p = float(input("Inserisci la percentuale tra 0 e 1 esclusi di dati da usare nel set di test: "))
        if p <= 0 or p >= 1:
            print("Valore non valido per la percentuale. Utilizzato il valore di default 0.25.")
            p = 0.25

        # Gestione della scelta dell'utente
        if choice == "1":
            splitter = Split(percentage=p)
            splits = splitter.holdout(X, Y)
            return splits
        if choice == "2":
            n = int(input("Inserisci un numero intero a partire da 1 per il numero di iterazioni dell'algoritmo: "))
            try:
                if n < 1:
                    raise ValueError
            except ValueError:
                print("Valore non valido per il numero di iterazioni: il valore deve essere un numero intero positivo. Utilizzato il valore di default 5.")
                n = 5
            splitter = Split(percentage=p, iterations=n)
            splits = splitter.random_subsampling(X, Y)
            return splits
        if choice == "3":
            n = int(input("Inserisci un numero intero a partire da 1 per il numero di iterazioni dell'algoritmo: "))
            try:
                if n < 1:
                    raise ValueError
            except ValueError:
                print("Valore non valido per il numero di iterazioni: il valore deve essere un numero intero positivo. Utilizzato il valore di default 5.")
                n = 5
            splitter = Split(percentage=p, iterations=n)
            splits = splitter.bootstrap(X, Y)
            return splits
        else:
            splitter = Split(percentage=p)
            splits = splitter.holdout(X, Y)
            print("Scelta non valida. Eseguito holdout")
            return splits

    @staticmethod
    def get_user_choice() -> str:
        """
        Chiede all'utente se vuole calcolare una o più metriche specifiche o tutte le metriche.

        return:
        --------
        str:
            "all" se l'utente vuole calcolare tutte le metriche.
        list:
            altrimenti una lista delle metriche selezionate.
        """
        print("Vuoi calcolare una o più metriche specifiche o tutte le metriche?")
        print("1. Calcola una o più metriche specifiche")
        print("2. Calcola tutte le metriche")
        choice = input("Inserisci il numero della tua scelta: ")

        if choice == "1":
            print("Scegli una o più metriche da calcolare:")
            print("1. Accuracy Rate")
            print("2. Error Rate")
            print("3. Sensitivity")
            print("4. Specificity")
            print("5. False Alarm Rate")
            print("6. Miss Rate")
            print("7. Geometric Mean")
            print("8. Area Under the Curve")

            metric_choices = input("Inserisci i numeri corrispondenti alle metriche desiderate, separate da virgole: ")

            metrics_dict = {
                "1": "Accuracy Rate",
                "2": "Error Rate",
                "3": "Sensitivity",
                "4": "Specificity",
                "5": "False Alarm Rate",
                "6": "Miss Rate",
                "7": "Geometric Mean",
                "8": "Area Under the Curve"
            }

            selected_metrics = [metrics_dict.get(choice.strip()) for choice in metric_choices.split(',') if choice.strip() in metrics_dict]
            return selected_metrics
        elif choice == "2":
            return "all"
        else:
            print("Nessuna metrica selezionata. Calcolo tutte le metriche.")
            return "all"

    @staticmethod
    def process_user_choice(user_choice) -> list:
        """
        Processa la scelta dell'utente.

        Parametri:
        ----------
        user_choice : str or list
            La scelta dell'utente riguardo le metriche da calcolare.

        Ritorna:
        --------
        list:
            Lista delle metriche da calcolare.
        """
        if user_choice == "all":
            user_choice = ['Accuracy Rate','Error Rate','Sensitivity','Specificity','False Alarm Rate','Miss Rate','Geometric Mean','Area Under the Curve']
        elif user_choice:
            user_choice = user_choice
        return user_choice

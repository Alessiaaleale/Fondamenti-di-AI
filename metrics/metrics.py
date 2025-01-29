import numpy as np

class MetricsCalculator:
    """
    Classe per calcolare le metriche di classificazione.
    """

    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
        self.tp, self.tn, self.fp, self.fn = confusion_matrix

    def accuracy_rate(self) -> float:
        """
        Calcola l'accuracy rate.
        """
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def error_rate(self) -> float:
        """
        Calcola l'error rate.
        """
        return 1 - self.accuracy_rate()

    def sensitivity(self) -> float:
        """
        Calcola la sensitivity.
        """
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0

    def specificity(self) -> float:
        """
        Calcola la specificity.
        """
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0

    def false_alarm_rate(self) -> float:
        """
        Calcola la false alarm rate.
        """
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0

    def miss_rate(self) -> float:
        """
        Calcola la miss rate.
        """
        return self.fn / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0

    def geometric_mean(self) -> float:
        """
        Calcola la geometric mean.
        """
        sens = self.sensitivity()
        spec = self.specificity()
        return np.sqrt(sens * spec)

    def auc(self) -> float:
        """
        Calcola l'area under the curve.
        """
        sens = self.sensitivity()
        spec = self.specificity()
        return (sens + spec) / 2

    def calculate_metrics(self, metrics=None)-> dict:
        """
        Calcola le metriche richieste.
        """
        if len(self.confusion_matrix) != 4:
            raise ValueError("La matrice di confusione deve contenere esattamente 4 valori: [TP, TN, FP, FN]")

        if any(x < 0 for x in [self.tp, self.tn, self.fp, self.fn]):
            raise ValueError("I valori della matrice di confusione non possono essere negativi.")

        all_metrics = {
            'Accuracy Rate': self.accuracy_rate(),
            'Error Rate': self.error_rate(),
            'Sensitivity': self.sensitivity(),
            'Specificity': self.specificity(),
            'False Alarm Rate': self.false_alarm_rate(),
            'Miss Rate': self.miss_rate(),
            'Geometric Mean': self.geometric_mean(),
            'Area Under the Curve': self.auc()
        }
        if metrics:
            invalid_metrics = [metric for metric in metrics if metric not in all_metrics]
            if invalid_metrics:
                raise ValueError(f"Metriche non valide: {', '.join(invalid_metrics)}. Le opzioni disponibili sono: {', '.join(all_metrics.keys())}")
            return {metric: all_metrics[metric] for metric in metrics}

        return all_metrics

    @staticmethod
    def get_user_choice()-> str:
        """
        Chiede all'utente se vuole calcolare una o più metriche specifiche o tutte le metriche.
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
    def process_user_choice(user_choice)-> list:
        """
        Processa la scelta dell'utente.
        """
        if user_choice == "all":
            user_choice = ['Accuracy Rate','Error Rate','Sensitivity','Specificity','False Alarm Rate','Miss Rate','Geometric Mean','Area Under the Curve']
        elif user_choice:
            user_choice = user_choice
        return user_choice
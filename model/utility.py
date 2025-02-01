import numpy as np
from metrics.metrics import MetricsCalculator
from knn import KNNClassifier

class classification_evaluation:
    def knn_metrics(k, splits,user_choice) -> dict:
        """
        Questa funzione estrae le tuple di test e train dalla lista degli split, derivante da holdout,
        random subsampling e bootstrap, e calcola le metriche richieste dall'utente per ogni split.
        Infine, calcola la media delle metriche per ogni split.

        :param k: numero di vicini
        :param splits: lista di tuple di test e train
        :param user_choice: lista di metriche scelte dall'utente
        """

        metrics_dict = {item: [] for item in user_choice}
        lista_metriche = []
        for i in range(len(splits)):
            ytest = splits[i][3]
            xtrain = splits[i][0]
            ytrain = splits[i][1]
            xtest = splits[i][2]

            knn_classifier = KNNClassifier(k)
            confusion_matrix = knn_classifier.calculate_confusion_matrix(ytest, knn_classifier.knn(xtrain, ytrain, xtest))
            calculator = MetricsCalculator(confusion_matrix)

            all_results = calculator.calculate_metrics(user_choice)
            lista_metriche.append(all_results)

            for metric, value in all_results.items():
                metrics_dict[metric].append(value)

        mean_metrics = {key: np.mean(values) for key, values in metrics_dict.items() if values}
        return mean_metrics
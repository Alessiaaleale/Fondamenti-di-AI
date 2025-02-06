import numpy as np
from metrics_results.metrics import MetricsCalculator
from model.knn import KNNClassifier
from metrics_results.results import ResultSaver

class classification_evaluation:
    def knn_metrics(k, splits, user_choice) -> dict:
        """
        Questa funzione estrae le tuple di test e train dalla lista degli split, derivante da holdout,
        random subsampling e bootstrap, e calcola le metriche richieste dall'utente per ogni split.
        Infine, calcola la media delle metriche per ogni split.

        Parametri
        ----------
        k : int
            Numero di vicini da considerare nell'algoritmo KNN.
        splits : list of tuples
            Lista di tuple contenenti i dati di test e train.
        user_choice : list of str
            Lista delle metriche scelte dall'utente da calcolare.

        Return
        -------
        dict
            Dizionario con le metriche calcolate per ciascuno split e la loro media.

        """

        metrics_dict = {item: [] for item in user_choice}
        lista_metriche = []

        for i in range(len(splits)):
            ytest = splits[i][3]
            xtrain = splits[i][0]
            ytrain = splits[i][1]
            xtest = splits[i][2]

            knn_classifier = KNNClassifier(k)
            ypred = knn_classifier.knn(xtrain, ytrain, xtest)
            confusion_matrix = knn_classifier.calculate_confusion_matrix(ytest, ypred)
            calculator = MetricsCalculator(confusion_matrix, ypred, ytest)

            all_results = calculator.calculate_metrics(user_choice)
            lista_metriche.append(all_results)

            for metric, value in all_results.items():
                metrics_dict[metric].append(value)

        mean_metrics = {key: np.mean(values) for key, values in metrics_dict.items() if values}

        # Salva il grafico dell'andamento delle metriche
        ResultSaver.save_plot(lista_metriche, user_choice, splits)

        # Salva le metriche in un file Excel
        ResultSaver.save_metrics_to_excel(lista_metriche, mean_metrics)
        
        return mean_metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ResultSaver:
    """
    Classe per salvare i risultati delle metriche in vari formati.
    """
    @staticmethod
    def save_plot(lista_metriche: list, user_choice: list, splits: list):
        """
        Salva il grafico dell'andamento delle metriche al crescere delle iterazioni.

        Parametri:
        ----------
        lista_metriche : list
            Lista delle metriche calcolate per ogni split.
        user_choice : list
            Lista delle metriche scelte dall'utente.
        splits : list
            Lista di split dei dati (iterazioni).
        """
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 6))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA500']

        for i, metric in enumerate(user_choice):
            metric_values = [split_metrics[metric] for split_metrics in lista_metriche if metric in split_metrics]
            plt.plot(metric_values, label=metric, marker='o', color=colors[i % len(colors)])

        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Metric value', fontsize=14)
        plt.title('Metrics trend over iterations', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(np.arange(len(splits)), np.arange(1, len(splits) + 1))

        plt.savefig('metrics_trend.png', dpi=300)
        plt.close()

    @staticmethod
    def save_metrics_to_excel(lista_metriche: list, mean_metrics: dict, filename: str = 'metrics.xlsx') -> str:
        """
        Salva le metriche in un file Excel.

        Parametri:
        ----------
        lista_metriche : list
            Lista delle metriche calcolate per ogni split.
        mean_metrics : dict
            Dizionario contenente le metriche medie.
        filename : str, optional
            Il nome del file Excel in cui salvare le metriche (default è 'metrics.xlsx').

        Ritorna:
        --------
        str:
            Il nome del file Excel in cui le metriche sono state salvate.
        """
        df = pd.DataFrame(lista_metriche)
        mean_metrics_df = pd.DataFrame([mean_metrics])
        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, index=False, sheet_name='Metrics')
            mean_metrics_df.to_excel(writer, index=False, sheet_name='Mean Metrics')
        return filename
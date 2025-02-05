import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ResultSaver:
    @staticmethod
    def save_plot(lista_metriche: list, user_choice: list, splits: list):
        """
        Salva il grafico dell'andamento delle metriche al crescere delle iterazioni.
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

    def save_metrics_to_excel(lista_metriche:list, mean_metrics:list, filename='metrics.xlsx'):
        """
        Salva le metriche in un file Excel.
        """
        df = pd.DataFrame(lista_metriche)
        mean_metrics_df = pd.DataFrame([mean_metrics])
        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, index=False, sheet_name='Metrics')
            mean_metrics_df.to_excel(writer, index=False, sheet_name='Mean Metrics')
        return filename
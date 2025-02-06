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
        user_choice_2 = [metric for metric in user_choice if metric != 'Area Under the Curve']
        for i, metric in enumerate(user_choice_2):
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

    @staticmethod
    def plot_roc_curve(fpr, tpr):
        """
        Plotta la curva ROC (Receiver Operating Characteristic).

        Parametri
        ----------
        fpr : array
            Array contenente i valori del False Positive Rate (tasso di falsi positivi).
        tpr : array
            Array contenente i valori del True Positive Rate (tasso di veri positivi).

        Return
        -------
        None
            La funzione non restituisce valori ma mostra un grafico ROC.

        """
        plt.style.use('ggplot')
        plt.figure(figsize=(6, 6))
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)

        # Disegna la curva ROC
        plt.plot(fpr, tpr, marker='o', linestyle='-')

        plt.fill_between(fpr, tpr, alpha=0.3, color='blue')
        plt.legend()
        plt.savefig("ROC.png", dpi=300)
        plt.close()
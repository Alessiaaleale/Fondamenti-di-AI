import unittest
from metrics_results.metrics import MetricsCalculator
import numpy as np

class TestMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.confusion_matrix = [50, 40, 10, 5]
        self.ypred = np.array([1, 0, 1, 0, 1])
        self.ytest = np.array([0, 1, 1, 0, 1])
        self.target_column = 1
        self.calculator = MetricsCalculator(self.confusion_matrix, self.ypred, self.ytest)

    def test_accuracy_rate(self):
        self.assertGreaterEqual(self.calculator.accuracy_rate(), 0)
        self.assertLessEqual(self.calculator.accuracy_rate(), 1)

    def test_error_rate(self):
        self.assertLessEqual(self.calculator.error_rate(), 1)
        self.assertGreaterEqual(self.calculator.error_rate(), 0)

    def test_sensitivity(self):
        self.assertGreaterEqual(self.calculator.sensitivity(), 0)
        self.assertLessEqual(self.calculator.sensitivity(), 1)

    def test_specificity(self):
        self.assertGreaterEqual(self.calculator.specificity(), 0)
        self.assertLessEqual(self.calculator.specificity(), 1)

    def test_false_alarm_rate(self):
        self.assertLessEqual(self.calculator.false_alarm_rate(), 1)
        self.assertGreaterEqual(self.calculator.false_alarm_rate(), 0)

    def test_miss_rate(self):
        self.assertLessEqual(self.calculator.miss_rate(), 1)
        self.assertGreaterEqual(self.calculator.miss_rate(), 0)

    def test_geometric_mean(self):
        self.assertGreaterEqual(self.calculator.geometric_mean(), 0)
        self.assertLessEqual(self.calculator.geometric_mean(), 1)

    def test_auc(self):
        self.assertGreaterEqual(self.calculator.auc(), 0)
        self.assertLessEqual(self.calculator.auc(), 1)

    def test_calculate_metrics_all(self):
        metrics_to_test = [
            'Accuracy Rate', 'Error Rate', 'Sensitivity', 'Specificity', 
            'False Alarm Rate', 'Miss Rate', 'Geometric Mean', 'Area Under the Curve'
        ]
        metrics = self.calculator.calculate_metrics(metrics_to_test)
        self.assertGreaterEqual(metrics['Accuracy Rate'], 0)
        self.assertLessEqual(metrics['Accuracy Rate'], 1)
        self.assertLessEqual(metrics['Error Rate'], 1)
        self.assertGreaterEqual(metrics['Error Rate'], 0)
        self.assertGreaterEqual(metrics['Sensitivity'], 0)
        self.assertLessEqual(metrics['Sensitivity'], 1)
        self.assertGreaterEqual(metrics['Specificity'], 0)
        self.assertLessEqual(metrics['Specificity'], 1)
        self.assertLessEqual(metrics['False Alarm Rate'], 1)
        self.assertGreaterEqual(metrics['False Alarm Rate'], 0)
        self.assertLessEqual(metrics['Miss Rate'], 1)
        self.assertGreaterEqual(metrics['Miss Rate'], 0)
        self.assertGreaterEqual(metrics['Geometric Mean'], 0)
        self.assertLessEqual(metrics['Geometric Mean'], 1)
        self.assertGreaterEqual(metrics['Area Under the Curve'], 0)
        self.assertLessEqual(metrics['Area Under the Curve'], 1)

    def test_calculate_metrics_specific(self):
        metrics = self.calculator.calculate_metrics(['Accuracy Rate', 'Sensitivity'])
        self.assertGreaterEqual(metrics['Accuracy Rate'], 0)
        self.assertLessEqual(metrics['Accuracy Rate'], 1)
        self.assertGreaterEqual(metrics['Sensitivity'], 0)
        self.assertLessEqual(metrics['Sensitivity'], 1)
        self.assertNotIn('Specificity', metrics)
        self.assertNotIn('Error Rate', metrics)
        self.assertNotIn('False Alarm Rate', metrics)
        self.assertNotIn('Miss Rate', metrics)
        self.assertNotIn('Geometric Mean', metrics)
        self.assertNotIn('Area Under the Curve', metrics)

    def test_invalid_confusion_matrix(self):
        with self.assertRaises(ValueError):
            MetricsCalculator([50, 40, 10], self.ypred, self.ytest)

    def test_negative_values_in_confusion_matrix(self):
        with self.assertRaises(ValueError):
            MetricsCalculator([50, -10, -5, 5], self.ypred, self.ytest)

    def test_division_by_zero(self):
        calculator = MetricsCalculator([0, 0, 0, 0], self.ypred, self.ytest)
        with self.assertRaises(ValueError):
            calculator.sensitivity()
        with self.assertRaises(ValueError):
            calculator.specificity()
        with self.assertRaises(ValueError):
            calculator.false_alarm_rate()
        with self.assertRaises(ValueError):
            calculator.miss_rate()
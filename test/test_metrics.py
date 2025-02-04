import unittest
from metrics_results.metrics import MetricsCalculator

class TestMetricsCalculator(unittest.TestCase):
    def test_accuracy_rate(self):
        calculator = MetricsCalculator([50, 40, 10, 5])
        self.assertGreaterEqual(calculator.accuracy_rate(), 0)
        self.assertLessEqual(calculator.accuracy_rate(), 1)

    def test_error_rate(self):
        calculator = MetricsCalculator([50, 40, 10, 5])
        self.assertLessEqual(calculator.error_rate(), 1)
        self.assertGreaterEqual(calculator.error_rate(), 0)

    def test_sensitivity(self):
        calculator = MetricsCalculator([50, 40, 10, 5])
        self.assertGreaterEqual(calculator.sensitivity(), 0)
        self.assertLessEqual(calculator.sensitivity(), 1)

    def test_specificity(self):
        calculator = MetricsCalculator([50, 40, 10, 5])
        self.assertGreaterEqual(calculator.specificity(), 0)
        self.assertLessEqual(calculator.specificity(), 1)

    def test_false_alarm_rate(self):
        calculator = MetricsCalculator([50, 40, 10, 5])
        self.assertLessEqual(calculator.false_alarm_rate(), 1)
        self.assertGreaterEqual(calculator.false_alarm_rate(), 0)

    def test_miss_rate(self):
        calculator = MetricsCalculator([50, 40, 10, 5])
        self.assertLessEqual(calculator.miss_rate(), 1)
        self.assertGreaterEqual(calculator.miss_rate(), 0)

    def test_geometric_mean(self):
        calculator = MetricsCalculator([50, 40, 10, 5])
        self.assertGreaterEqual(calculator.geometric_mean(), 0)
        self.assertLessEqual(calculator.geometric_mean(), 1)

    def test_auc(self):
        calculator = MetricsCalculator([50, 40, 10, 5])
        self.assertGreaterEqual(calculator.auc(), 0)
        self.assertLessEqual(calculator.auc(), 1)

    def test_calculate_metrics_all(self):
        calculator = MetricsCalculator([50, 40, 10, 5])
        metrics = calculator.calculate_metrics()
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
        calculator = MetricsCalculator([50, 40, 10, 5])
        metrics = calculator.calculate_metrics(['Accuracy Rate', 'Sensitivity'])
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
            MetricsCalculator([50, 40, 10])

    def test_negative_values_in_confusion_matrix(self):
        with self.assertRaises(ValueError):
            MetricsCalculator([50, -10, -5, 5])

    def test_division_by_zero(self):
        calculator = MetricsCalculator([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            calculator.sensitivity()
        with self.assertRaises(ValueError):
            calculator.specificity()
        with self.assertRaises(ValueError):
            calculator.false_alarm_rate()
        with self.assertRaises(ValueError):
            calculator.miss_rate()
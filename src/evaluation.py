import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Klasa odpowiedzialna za ocenę wyników modelu.
    """

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
        """
        Oblicza metryki i generuje raport wizualny na podstawie predykcji.
        """
        logger.info(f"Evaluating: {model_name}")

        # Metryki
        metrics = self._calculate_metrics(y_true, y_pred)

        logger.info(
            f"{model_name} results: RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}, R2 = {metrics['r2']:.4f}")

        # 2. Wizualizacja
        self._plot_diagnostics(y_true, y_pred, model_name)

        metrics['name'] = model_name
        return metrics

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def _plot_diagnostics(self, y_true, y_pred, name: str):
        plt.figure(figsize=(12, 5))

        # Scatter plot
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.1, color='teal', edgecolor=None)
        max_val = max(y_true.max(), y_pred.max())
        # Dodajemy linię "Ideal Fit"
        plt.plot([0, max_val], [0, max_val], '--r', linewidth=2, label='Ideal Fit')
        plt.xlabel('Prawdziwa Popularność')
        plt.ylabel('Przewidziana Popularność')
        plt.title(f'{name}: Actual vs Predicted')
        plt.legend()

        # Histogram błędów
        residuals = y_true - y_pred
        plt.subplot(1, 2, 2)
        sns.histplot(residuals, bins=50, kde=True, color='purple', linewidth=0)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'{name}: Rozkład Błędów (Residuals)')
        plt.xlabel('Błąd')

        plt.tight_layout()
        plt.show()

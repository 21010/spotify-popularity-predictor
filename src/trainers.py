import logging
import time
from typing import Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Klasa odpowiedzialna za proces trenowania modelu.
    """

    def train(self, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Trenuje przekazany model na danych treningowych.

        Args:
            model: Obiekt modelu (musi posiadać metodę .fit)
            X_train: Cechy treningowe
            y_train: Zmienna celu

        Returns:
            Wytrenowany obiekt modelu.
        """
        model_name = model.__class__.__name__
        logger.info(f"Starting training for: {model_name}")

        start_time = time.time()

        try:
            # Trenujemy model
            model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            raise e

        elapsed_time = time.time() - start_time
        logger.info(f"Training completed for {model_name} in {elapsed_time:.2f} seconds.")

        return model

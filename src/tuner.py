import logging
from typing import Any

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


logger = logging.getLogger(__name__)


class ModelTuner:
    """
    Klasa odpowiedzialna za znalezienie najlepszych hiperparametrów modelu.
    """

    def __init__(self, n_iter: int = 20, cv: int = 3):
        self.n_iter = n_iter  # liczba kombinacji
        self.cv = cv  # ilość podziałów danych

    def tune(self, X, y) -> dict[str, Any]:
        """
        Uruchamia poszukiwanie najlepszych parametrów.
        """
        logger.info(f"Rozpoczynanie tuningu (iteracje: {self.n_iter}, CV: {self.cv})... To może chwilę potrwać.")

        # Definicja przestrzeni poszukiwań (Grid)
        param_dist = {
            'n_estimators': [500, 1000, 1500],  # liczba drzew
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # szybkość uczenia
            'max_depth': [4, 6, 8, 10],  # głębokość drzewa
            'subsample': [0.7, 0.8, 0.9, 1.0],  # Jaki % próbek brać do każdego drzewa
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # Jaki % kolumn brać do każdego drzewa
            'min_child_weight': [1, 3, 5]  # Ochrona przed overfittingiem
        }

        # Model bazowy
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)

        # Konfiguracja przeszukiwania
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            scoring='neg_root_mean_squared_error',  # Optymalizujemy pod RMSE
            cv=self.cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X, y)

        logger.info(f"Najlepsze parametry znalezione: {random_search.best_params_}")
        logger.info(f"Najlepszy wynik (RMSE z CV): {-random_search.best_score_:.4f}")

        return random_search.best_params_

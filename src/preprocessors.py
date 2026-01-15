import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


logger = logging.getLogger(__name__)


class DataPreprocessor(ABC):
    """
    Abstrakcja (Interfejs) dla preprocessora.
    Definiuje kontrakt: wejście DataFrame -> wyjście Macierze NumPy (X_train, X_test, y_train, y_test).
    """

    @abstractmethod
    def process(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        pass


class SpotifyPipelinePreprocessor(DataPreprocessor):
    """
    Implementacja preprocessora wykorzystująca Scikit-Learn Pipeline.
    """

    def __init__(self, target_col: str = 'popularity', test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []

    def process(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Starting data preprocessing pipeline...")

        # Sprawdzanie, czy kolumna celu istnieje w DataFrame
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")

        # Podział na cechy i cel
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Podział na zbiór treningowy i testowy
        logger.info(f"Splitting data into train/test sets (test_size={self.test_size}).")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Lista kolumn numerycznych i kategorycznych
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
        logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")

        # Definicja Pipeline'ów dla typów danych

        # Pipeline Numeryczny: Uzupełnij braki medianą -> Skalowanie
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline Kategoryczny: Uzupełnij braki najczęstszą wartością -> One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Łączenie Pipeline'ów w ColumnTransformer
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            verbose_feature_names_out=False
        )

        # Uczenie i transformacja danych
        logger.info("Fitting transformers on X_train...")
        X_train_processed = self.pipeline.fit_transform(X_train)

        logger.info("Transforming X_test...")
        X_test_processed = self.pipeline.transform(X_test)

        # Wyciągnięcie nazw kolumn po transformacji
        try:
            self.feature_names = self._extract_feature_names()
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")

        logger.info(f"Preprocessing finished. Final X_train shape: {X_train_processed.shape}")

        # Zwrócenie danych
        return X_train_processed, X_test_processed, y_train.values, y_test.values

    def get_feature_names(self) -> List[str]:
        """Zwraca listę nazw kolumn po transformacji."""
        if not self.feature_names:
            logger.warning("Feature names are empty. Did you run process()?")
        return self.feature_names

    def transform_new_data(self, df: pd.DataFrame):
        """
        Przetwarza nowe dane, używając wytrenowanego pipeline'u.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline nie został wytrenowany! Uruchom najpierw process() na danych treningowych.")

        # Zwraca tylko X (macierz cech), bo dla nowych danych nie ma y (targetu)
        return self.pipeline.transform(df)

    def _extract_feature_names(self) -> List[str]:
        """Metoda pomocnicza do wyciągania nazw z ColumnTransformera."""
        if hasattr(self.pipeline, 'get_feature_names_out'):
            return list(self.pipeline.get_feature_names_out())
        else:
            # Fallback dla starszych wersji scikit-learn
            return []

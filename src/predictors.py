import logging
import os.path
import argparse

import pandas as pd
from src.serializers import ModelSerializer


logger = logging.getLogger(__name__)


class SpotifyPredictor:
    """
    Klasa wrapper służąca do wykonywania predykcji na nowych danych.
    """

    def __init__(self, model_path: str, preprocessor_path: str):

        self.serializer = ModelSerializer()
        self.model = self.serializer.load(model_path)
        self.preprocessor = self.serializer.load(preprocessor_path)

    def predict(self, song_data: dict) -> float:
        """
        Przyjmuje słownik z danymi piosenki i zwraca przewidywaną popularność (0-100).
        """

        # Przygotowanie DataFrame
        df = pd.DataFrame([song_data])

        # Preprocessing-przygotowanie zbioru danych
        X = self.preprocessor.transform_new_data(df)

        # Predykcja
        prediction = self.model.predict(X)

        return float(prediction[0])

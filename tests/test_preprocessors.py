import numpy as np
from src.preprocessors import SpotifyPipelinePreprocessor


def test_preprocessor_splitting(sample_clean_data):
    preprocessor = SpotifyPipelinePreprocessor(target_col='popularity', test_size=0.5)
    X_train, X_test, y_train, y_test = preprocessor.process(sample_clean_data)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_preprocessor_encoding_and_scaling(sample_clean_data):
    preprocessor = SpotifyPipelinePreprocessor(target_col='popularity', test_size=0.2)
    X_train, _, _, _ = preprocessor.process(sample_clean_data)

    feature_names = preprocessor.get_feature_names()

    # Sprawdzenie, czy gatunki zostały zamienione na kolumny (One-Hot)
    genre_cols = [col for col in feature_names if 'track_genre' in col]
    assert len(genre_cols) >= 3

    # Sprawdzenie, czy X_train jest macierzą numpy
    assert isinstance(X_train, np.ndarray)

    # Sprawdzenie, czy wartości numeryczne zostały przeskalowane (nie są oryginalnymi wartościami)
    # Wybieramy kolumnę 'loudness' jako przykład
    loudness_idx = feature_names.index('loudness')
    # Sprawdzamy, czy średnia jest bliska 0 i odchylenie standardowe bliskie 1
    # Używamy tolerancji ze względu na małą liczbę próbek
    assert np.isclose(X_train[:, loudness_idx].mean(), 0, atol=0.5)
    assert np.isclose(X_train[:, loudness_idx].std(), 1, atol=0.5)


def test_transform_new_data(sample_clean_data):
    """Testuje metodę używaną w predictors.py"""
    preprocessor = SpotifyPipelinePreprocessor(target_col='popularity', test_size=0.2)

    # Trenujemy preprocessor na danych treningowych
    preprocessor.process(sample_clean_data)

    # Przykładowe dane testowe
    single_row = sample_clean_data.drop(columns=['popularity']).iloc[[0]]

    # Przekształcamy dane testowe za pomocą preprocessor'a
    X_new = preprocessor.transform_new_data(single_row)

    # Sprawdzamy kształt macierzy
    assert X_new.shape[0] == 1

    # Sprawdzamy czy liczba kolumn zgadza się z liczbą cech po transformacji
    assert X_new.shape[1] == len(preprocessor.get_feature_names())
import pytest
import os
from src.predictors import SpotifyPredictor

MODEL = 'spotify-xgb-model_v1.joblib'
PREPROCESSOR = 'spotify-preprocessor_v1.joblib'
MODEL_EXISTS = os.path.exists(f'models/{MODEL}')


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model v1 nie został jeszcze wytrenowany (brak plików w models/)")
def test_inference_end_to_end_demo():

    # Definiujemy Hit radiowy
    hit_song = {
        'track_id': 'test1', 'artists': 'Star', 'album_name': 'Hit Album', 'track_name': 'Hit',
        'explicit': True,
        'danceability': 0.85,
        'energy': 0.90,
        'key': 1,
        'loudness': -3.0,
        'mode': 1,
        'speechiness': 0.05,
        'acousticness': 0.01,
        'instrumentalness': 0.0,
        'liveness': 0.1,
        'valence': 0.9,
        'tempo': 128.0,
        'duration_ms': 200000,  # 3:20
        'time_signature': 4,
        'track_genre': 'rock'
    }

    # Definiujemy "niszę" (Muzyka relaksacyjna/klasyczna)
    niche_song = hit_song.copy()
    niche_song['energy'] = 0.2
    niche_song['loudness'] = -15.0
    niche_song['tempo'] = 60.0
    niche_song['track_genre'] = 'classical'  # Gatunek: Classical

    # Inicjalizacja predyktora
    predictor = SpotifyPredictor(MODEL, PREPROCESSOR)

    # Predykcje
    score_hit = predictor.predict(hit_song)
    score_niche = predictor.predict(niche_song)

    # Wyniki
    print(f"1. Utwór POP (Hit potential): {score_hit:.2f} / 100")
    print(f"2. Utwór CLASSICAL (Niche):   {score_niche:.2f} / 100")

    # Sprawdzamy, czy wyniki są liczbami z zakresu 0-100
    assert 0 <= score_hit <= 100
    assert 0 <= score_niche <= 100

    # Sprawdzamy, czy model "rozumie" różnicę.
    assert score_hit > score_niche, "Model powinien ocenić Pop wyżej niż Classical w tym kontekście"

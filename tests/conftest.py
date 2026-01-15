import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_raw_data():
    """
    Tworzy DataFrame imitujący surowe dane ze Spotify zgodnie z pełną specyfikacją.
    Zawiera 5 wierszy, w tym jeden duplikat (id1).
    """
    data = {
        # Metadata (Text / ID)
        'track_id': ['id1', 'id2', 'id3', 'id1', 'id4'],  # id1 to duplikat
        'artists': ['Artist A', 'Artist B;Artist C', 'Artist D', 'Artist A', 'Artist E'],
        'album_name': ['Album 1', 'Album 2', 'Album 3', 'Album 1', 'Album 4'],
        'track_name': ['Hit Song', 'Rock Anthem', 'Jazz Tune', 'Hit Song', 'Symphony No. 5'],
        'track_genre': ['pop', 'rock', 'jazz', 'pop', 'classical'],

        # Target
        'popularity': [80, 60, 20, 80, 40],  # 0-100

        # Boolean
        'explicit': [False, True, False, False, False],

        # Time & Structure
        'duration_ms': [210000, 180000, 240000, 210000, 300000],
        'time_signature': [4, 4, 3, 4, 5],  # 3-7
        'key': [0, 2, 5, 0, 9],  # 0-11 or -1
        'mode': [1, 0, 1, 1, 1],  # 0 or 1

        # Audio Features (Floats 0.0 - 1.0 or similar)
        'danceability': [0.8, 0.5, 0.4, 0.8, 0.1],
        'energy': [0.9, 0.8, 0.4, 0.9, 0.2],
        'loudness': [-5.0, -4.5, -12.0, -5.0, -20.0],  # dB (negative)
        'speechiness': [0.05, 0.04, 0.05, 0.05, 0.03],
        'acousticness': [0.1, 0.01, 0.8, 0.1, 0.95],
        'instrumentalness': [0.0, 0.002, 0.7, 0.0, 0.9],
        'liveness': [0.2, 0.1, 0.15, 0.2, 0.1],
        'valence': [0.9, 0.4, 0.3, 0.9, 0.1],
        'tempo': [120.0, 140.0, 90.0, 120.0, 70.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_clean_data(sample_raw_data):
    """
    Zwraca dane potencjalnie wyczyszczone (bez duplikatów i metadanych).
    Symuluje wynik działania SpotifyDataCleaner.clean().
    """
    # 1. Usuń duplikaty
    df = sample_raw_data.drop_duplicates(subset=['track_id'])

    # 2. Usuń kolumny tekstowe/metadata, których nie chcemy w modelu
    cols_to_drop = ['track_id', 'artists', 'album_name', 'track_name']
    df = df.drop(columns=cols_to_drop)

    return df
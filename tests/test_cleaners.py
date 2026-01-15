from src.cleaners import SpotifyDataCleaner


def test_cleaner_removes_duplicates(sample_raw_data):
    cleaner = SpotifyDataCleaner()
    df_clean = cleaner.clean(sample_raw_data)

    assert len(df_clean) == 4


def test_cleaner_drops_metadata_columns(sample_raw_data):
    cleaner = SpotifyDataCleaner()
    df_clean = cleaner.clean(sample_raw_data)

    forbidden_cols = ['track_id', 'track_name', 'artists', 'album_name']

    for col in forbidden_cols:
        assert col not in df_clean.columns


def test_cleaner_keeps_model_columns(sample_raw_data):
    cleaner = SpotifyDataCleaner()
    df_clean = cleaner.clean(sample_raw_data)

    required_cols = [
        'popularity', 'duration_ms', 'track_genre', 'explicit',
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]
    for col in required_cols:
        assert col in df_clean.columns

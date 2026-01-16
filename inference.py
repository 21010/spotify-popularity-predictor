import argparse
import logging
import sys
from src.predictors import SpotifyPredictor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Przykładowe dane (Szablon utworu)
SAMPLE_SONG = {
    'track_id': 'new_song_001',
    'artists': 'Unknown Artist',
    'album_name': 'New Album',
    'track_name': 'Demo Track',
    'explicit': False,
    'danceability': 0.70,
    'energy': 0.80,
    'key': 5,
    'loudness': -5.5,
    'mode': 1,
    'speechiness': 0.04,
    'acousticness': 0.10,
    'instrumentalness': 0.00,
    'liveness': 0.15,
    'valence': 0.65,
    'tempo': 120.0,
    'duration_ms': 210000,
    'time_signature': 4,
    'track_genre': 'pop'
}


def str2bool(v):
    """Pomocnicza funkcja do obsługi wartości logicznych w argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Oczekiwano wartości boolean (True/False).')


def main():
    # Konfiguracja parsera argumentów
    parser = argparse.ArgumentParser(description="Spotify Popularity Inference CLI")

    # Argumenty
    parser.add_argument('--version', type=str, default='v1', help='Wersja modelu (domyślnie: v1)')
    parser.add_argument('--genre', type=str, default='pop', help='Gatunek utworu (domyślnie: pop)')

    # Automatyczne generowanie argumentów na podstawie słownika SAMPLE_SONG
    ignored_keys = ['track_id', 'artists', 'album_name', 'track_name', 'track_genre']
    for key, val in SAMPLE_SONG.items():
        if key in ignored_keys:
            continue

        val_type = type(val)

        if val_type == bool:
            parser.add_argument(f'--{key}', type=str2bool, default=val, help=f'(bool) Domyślnie: {val}')
        else:
            parser.add_argument(f'--{key}', type=val_type, default=val, help=f'({val_type.__name__}) Domyślnie: {val}')

    args = parser.parse_args()

    try:
        # Aktualizacja parametrów utworu
        current_song = SAMPLE_SONG.copy()
        current_song['track_genre'] = args.genre

        args_dict = vars(args)
        for key in current_song.keys():
            if key in args_dict and key not in ignored_keys:
                current_song[key] = args_dict[key]

        logger.info(f"Przyjęte parametry utworu: {current_song}")

        # Inicjalizacja predyktora
        model_file = f"spotify-xgb-model_{args.version}.joblib"
        preprocessor_file = f"spotify-preprocessor_{args.version}.joblib"

        predictor = SpotifyPredictor(model_file, preprocessor_file)

        logger.info(f"Testowanie utworu. Gatunek: '{args.genre}'. Wersja modelu: {args.version}")

        # Wykonanie predykcji
        score = predictor.predict(current_song)

        # Wyświetlenie wyniku
        print(f"WYNIK PREDYKCJI POPULARNOŚCI")
        print(f"Gatunek:      {args.genre.upper()}")
        print(f"Popularność:  {score:.2f} / 100")
        print()

    except FileNotFoundError:
        logger.error(
            f"Nie znaleziono plików modelu dla wersji '{args.version}'. Upewnij się, że pliki .joblib istnieją.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Wystąpił nieoczekiwany błąd: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
import os
import joblib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelSerializer:
    """
    Klasa odpowiedzialna za zapisywanie i odczytywanie artefaktów (modeli, preprocessorów).
    """

    def __init__(self, base_dir: str = 'models'):
        self.base_dir = base_dir
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        """Tworzy katalog na modele, jeśli nie istnieje."""
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, obj: Any, filename: str) -> str:
        """
        Serializuje (zapisuje) obiekt do pliku.

        Args:
            obj: Obiekt do zapisania (np. model, preprocessor).
            filename: Nazwa pliku (np. 'model.joblib').

        Returns:
            Ścieżka do zapisanego pliku.
        """
        file_path = os.path.join(self.base_dir, filename)
        try:
            joblib.dump(obj, file_path)
            logger.info(f"Zapisano pomyślnie: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania {filename}: {e}")
            raise

    def load(self, filename: str) -> Any:
        """
        Deserializuje (wczytuje) obiekt z pliku.
        """
        file_path = os.path.join(self.base_dir, filename)
        if not os.path.exists(file_path):
            error_msg = f"Plik nie istnieje: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            obj = joblib.load(file_path)
            logger.info(f"Wczytano pomyślnie: {file_path}")
            return obj
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania {filename}: {e}")
            raise

import logging
from abc import ABC, abstractmethod

import pandas as pd


logger = logging.getLogger(__name__)


class DataCleaner(ABC):

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class SpotifyDataCleaner(DataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("\nStarting data cleaning.")
        df = df.copy()
        initial_shape = df.shape
        logger.info(f"\nInitial shape: {initial_shape}")

        df = self._drop_unnecessary_columns(df)
        df = self._remove_duplicates(df)
        df = self._drop_high_cardinality_columns(df)
        df = self._handle_missing_values(df)
        df = self._convert_data_types(df)

        logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        logger.info(f"\n(Removed {initial_shape[0] - df.shape[0]} rows)")
        logger.info(f"\nShape after cleaning: {df.shape}")

        return df

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Usuwa kolumny techniczne, nieprzydatne w modelowaniu."""
        cols_to_drop = ['Unnamed: 0']
        existing_cols = [c for c in cols_to_drop if c in df.columns]

        if existing_cols:
            logger.info(f"Dropping columns: {existing_cols}")
            return df.drop(columns=existing_cols)

        return df

    def _drop_high_cardinality_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = ['track_id','track_name', 'album_name', 'artists', 'Unnamed: 0']
        existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]

        if existing_cols_to_drop:
            logger.info(f"Dropping high-cardinality/ID columns: {existing_cols_to_drop}")
            df = df.drop(columns=existing_cols_to_drop)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Usuwa duplikaty. Ten sam utwór (track_id) może być na różnych albumach.
        Zostawiamy tylko pierwsze wystąpienie, aby uniknąć Data Leakage.
        """
        if 'track_id' in df.columns:
            duplicates = df.duplicated(subset=['track_id']).sum()
            if duplicates > 0:
                logger.info(f"Found {duplicates} duplicates based on 'track_id'. Removing...")
                df = df.drop_duplicates(subset=['track_id'], keep='first')

                # Po usunięciu duplikatów, track_id nie jest już potrzebne jako cecha
                df = df.drop(columns=['track_id'])
        else:
            # Fallback dla standardowych duplikatów
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                logger.info(f"Found {duplicates} exact duplicates. Removing...")
                df = df.drop_duplicates()

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Obsługa brakujących danych (NaN)."""
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values. Dropping rows with NaNs.")
            return df.dropna()

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optymalizacja typów danych (np. object -> category)."""
        if 'track_genre' in df.columns:
            df['track_genre'] = df['track_genre'].astype('category')
        if 'explicit' in df.columns:
            df['explicit'] = df['explicit'].astype(int)

        return df

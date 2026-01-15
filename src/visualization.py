import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SpotifyVisualizer:

    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Ustawienia estetyczne dla wszystkich wykresów
        # sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def plot_target_distribution(self, target_col: str = 'popularity'):
        """
        Analiza rozkładu zmiennej celu.
        """
        plt.figure()
        sns.histplot(self.df[target_col], bins=50, kde=True, color='purple')
        plt.title(f'Rozkład zmiennej celu: {target_col}')
        plt.xlabel('Wartość')
        plt.ylabel('Liczebność')
        plt.show()

        # Podstawowe statystyki
        stats = self.df[target_col].describe()
        logger.info(f"Statystyki celu:\n{stats}")

    def plot_correlation_matrix(self):
        """
        Generuje heatmapę korelacji, aby zobrazować zależności między cechami.
        """
        # Wybieramy tylko kolumny numeryczne do korelacji
        numeric_df: pd.DataFrame = self.df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            logger.warning("Brak kolumn numerycznych do analizy korelacji.")
            return

        plt.figure(figsize=(12, 10))
        corr = numeric_df.corr(method='spearman')

        mask = np.triu(np.ones_like(corr, dtype=bool))  # Ukrywa górny trójkąt (duplikaty)

        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmax=1, vmin=-1, linewidths=0.5)
        plt.title('Macierz Korelacji (Spearman)')
        plt.show()

    def plot_features_vs_target(self, target_col: str = 'popularity', features: list = None, sample_size: int = 1000):
        """
        Tworzy wykresy punktowe (Scatter plots) dla wybranych cech względem celu.
        Pozwala ocenić liniowość relacji.
        """
        if features is None:
            # Domyślnie najważniejsze cechy audio
            features = ['danceability', 'energy', 'loudness', 'acousticness', 'valence']

        for feature in features:
            if feature not in self.df.columns:
                continue

            plt.figure(figsize=(10, 5))
            # Próbkowanie danych dla lepszej czytelności wykresu
            if len(self.df) > sample_size:
                sample_df = self.df.sample(n=sample_size, random_state=42)
            else:
                sample_df = self.df

            sns.regplot(x=feature, y=target_col, data=sample_df, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'})
            plt.title(f'{feature} vs {target_col}')
            plt.show()

    def plot_genre_popularity(self, top_n: int = 20):
        """
        Analiza wpływu gatunku na popularność (Boxplot).
        """
        if 'track_genre' not in self.df.columns:
            return

        # Obliczamy średnią popularność dla gatunków i sortujemy
        order = self.df.groupby('track_genre', observed=False)['popularity'].mean().sort_values(ascending=False).index[:top_n]

        plt.figure(figsize=(14, 8))
        sns.boxplot(x='track_genre', y='popularity', data=self.df, order=order, palette='viridis', hue='track_genre', legend=False)
        plt.title(f'Top {top_n} Gatunków wg Popularności')
        plt.xticks(rotation=45)
        plt.show()

    def plot_audio_features_distribution(self, features: list = None):
        """
        Wykresy rozkładu dla kluczowych cech audio.
        """
        if features is None:
            features = [
                'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo'
            ]

        num_features = len(features)
        if num_features == 0:
            logger.warning("Brak cech audio do wizualizacji.")
            return

        # Obliczanie optymalnej siatki dla sub-wykresów
        n_cols = 3
        n_rows = (num_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()  # Spłaszczenie tablicy osi dla łatwiejszej iteracji

        for i, feature in enumerate(features):
            if feature in self.df.columns:
                sns.histplot(self.df[feature], bins=30, kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f'Rozkład: {feature}')
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')
            else:
                logger.warning(f"Cecha '{feature}' nie istnieje w zbiorze danych.")
                # Usuń puste sub-wykresy
                fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def plot_duration_analysis(self, target_col: str = 'popularity', sample_size: int = 1000):
        """
        Analiza wpływu długości utworu na popularność.
        Zawiera konwersję ms -> minuty oraz filtrację ekstremalnych outlierów (>15 min) dla czytelności.
        """
        if 'duration_ms' not in self.df.columns:
            logger.warning("Brak kolumny 'duration_ms'. Pomijam analizę czasu trwania.")
            return

        df_viz = self.df.copy()
        df_viz['duration_min'] = df_viz['duration_ms'] / 60000

        # Filtrujemy audiobooki i sety DJ-skie (> 10 min) dla czytelności wykresu
        df_filtered = df_viz[df_viz['duration_min'] < 10]

        # Przygotowujemy próbkę
        if len(df_filtered) > sample_size:
            sample_df = df_filtered.sample(n=sample_size, random_state=42)
        else:
            sample_df = df_filtered

        # Generujemy wykres
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='duration_min', y=target_col, data=sample_df, alpha=0.1, color='teal', edgecolor=None)
        # Dodajemy linię "Radio Edit Standard" (ok. 3:30 min)
        plt.axvline(x=3.5, color='red', linestyle='--', linewidth=1.5, label='Radio Edit (~3.5 min)')

        plt.title('Długość utworu vs Popularność')
        plt.xlabel('Czas trwania (minuty)')
        plt.ylabel('Popularność')
        plt.legend()
        plt.show()

    def plot_explicit_content_impact(self, sample_size: int = 1000):
        """
        Analiza wpływu treści wulgarnych (Explicit) na popularność.
        """
        if 'explicit' not in self.df.columns:
            logger.warning("Brak kolumny 'explicit'. Pomijam analizę.")
            return

        # Przygotowujemy próbkę
        if len(self.df) > sample_size:
            sample_df = self.df.sample(n=sample_size, random_state=42)
        else:
            sample_df = self.df

        # Generujemy wykres boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='explicit', y='popularity', data=sample_df, palette='Set2', hue='explicit', legend=False)

        plt.title('Wpływ treści Explicit na popularność')
        plt.xlabel('Czy utwór jest Explicit?')
        plt.ylabel('Popularność')
        plt.show()

        # Logujemy statystyki
        mean_vals = self.df.groupby('explicit', observed=False)['popularity'].mean()
        logger.info(f"Średnia popularność - Not Explicit: {mean_vals.get(False, 0):.2f}")
        logger.info(f"Średnia popularność - Explicit: {mean_vals.get(True, 0):.2f}")

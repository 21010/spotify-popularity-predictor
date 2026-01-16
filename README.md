# Projekt ML: Predykcja Popularności Utworów Spotify

## O projekcie

Celem projektu jest zbudowanie modelu uczenia maszynowego, który na podstawie cech audio (np. tempo, głośność, akustyczność)
oraz gatunku muzycznego przewiduje indeks popularności (0-100) utworu na Spotify.

Projekt realizuje pełny cykl życia aplikacji ML:
1. Exploratory Data Analysis (Analiza EDA) - zrozumienie danych i relacji biznesowych.
2. Feature Engineering - transformacja danych (One-Hot Encoding, Scaling).
3. Modelowanie - porównanie algorytmów (Baseline vs Linear Regression vs XGBoost).
4. Tuning - optymalizacja hiperparametrów w celu uzyskania optymalnych wyników.
5. Inżynieria Oprogramowania - czysty kod (OOP), testy jednostkowe, pipeline treningowy.

## Wyniki i Wnioski Biznesowe

Po przetestowaniu kilku podejść, najlepsze wyniki osiągnął model XGBoost, radząc sobie znacznie lepiej z nieliniowymi zależnościami niż klasyczna regresja.

| Model                 | RMSE (Błąd)   | R2 Score   | Opis                                    |
|:----------------------|:--------------|:-----------|:----------------------------------------|
| **Baseline (Mean)**   | 20.44         | ~0.00      | Punkt odniesienia (zgadywanie średniej) |
| **Linear Regression** | 16.84         | 0.32       | Wykrywa proste trendy, ale gubi niuanse |
| **XGBoost (Final)**   | **14.90**     | **0.47**   | Najlepsza precyzja i stabilność         |

**Kluczowe wnioski:**
* Najsilniejszym predyktorem sukcesu jest gatunek muzyczny (np. Pop vs Grindcore). Determinuje on "sufit" popularności.
* Błąd średni (MAE) wynosi ok. 10 punktów. Model nie jest "magiczną kulą", ale doskonale sprawdza się jako narzędzie wspierające decyzje, odfiltrowując utwory o niskim potencjale technicznym.

## Wykorzystane Narzędzia

* Język: Python 3.x
* Biblioteki: XGBoost, Scikit-Learn, Pandas, NumPy
* Narzędzia: Joblib (serializacja), Pytest (testy), Matplotlib/Seaborn (wizualizacja)
* Format danych: CSV (źródło), Parquet (dane pośrednie)

## Instrukcja uruchomienia projektu

### 1. Instalacja zależności

1. (Krok opcjonalny, jeśli brak zainstalowanego `uv` w systemie.)

    ```bash
    pip install uv
    ```

2. Instalacja zależności

    ```bash
    uv sync 
    ```
3. Uruchomienie pipeline'u treningowego (pobranie danych, czyszczenie, trenowanie i zapisanie modelu w `models`)

    ```bash
    uv run ./main.py
    ```
4. Symulacja predykcji dla nowego utworu.

    ```bash
    uv run ./inference.py --genre pop --tempo 160 --energy 0.95 --loudness -2.0
    uv run ./inference.py --genre rock --valence 0.1 --danceability 0.2
    ```
5. Uruchomienie testów 

    Sprawdzenie spójności danych i poprawności transformacji.

    ```bash
    pytest
    ```

## Struktura Projektu
Projekt został zaprojektowany zgodnie z zasadami SOLID i Clean Code.

├── data/                  # Dane surowe i przetworzone (ignorowane przez git)
├── models/                # Zapisane modele (.joblib)
├── notebooks/             # Notatniki Jupyter (Eksperymenty)
│   ├── 1. Analiza EDA.ipynb
│   ├── 2. Feature Engineering.ipynb
│   ├── 3. Modeling.ipynb
│   └── 4. Tuning.ipynb
├── src/                   # Kod źródłowy (Logika biznesowa)
│   ├── cleaners.py        # Czyszczenie danych
│   ├── loaders.py         # Pobieranie danych
│   ├── preprocessors.py   # Transformacje (Pipeline)
│   ├── trainers.py        # Logika treningu
│   ├── evaluation.py      # Metryki i wykresy
│   ├── serializers.py     # Zapis/Odczyt modeli
│   └── predictor.py       # Klasa do inferencji
│   └── visualization.py   # Wizualizacje
├── tests/                 # Testy jednostkowe i integracyjne
├── main.py                # Orkiestrator treningu (CLI)
├── inference.py           # Skrypt do predykcji (CLI)
├── tune_pipeline.py       # Skrypt do szukania hiperparametrów
└── pyproject.toml         # `uv` konfiguracja zależności

## Plan rozwoju
Uzyskane wyniki wskazują, że model może być wykorzystany jako narzędzie wspierające decyzje, ale nie pozwala w pełni przewidzieć, który utwór ma potencjał na wysoką popularność.
W dalszym rozwoju rozwiązania należy rozważyć:
* wzbogacenie danych o dane o zasięgach w social mediach, budżet marketingowy i inne specyficzne dla branży muzycznej.
* powiększenie zbioru danych

Aby przygotować rozwiązanie do uruchomienia:
* Wystawienie modelu jako REST API (FastAPI).
* Konteneryzacja aplikacji (Docker).
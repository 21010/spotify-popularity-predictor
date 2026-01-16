import logging
import xgboost as xgb
from src.loaders import DataLoaderFactory
from src.cleaners import SpotifyDataCleaner
from src.preprocessors import SpotifyPipelinePreprocessor
from src.tuner import ModelTuner
from src.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_tuning():
    # Wczytanie i czyszczenie danych
    data_source = "hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv"
    loader = DataLoaderFactory.get_loader(data_source)
    cleaner = SpotifyDataCleaner()
    df_clean = cleaner.clean(loader.load())

    # Preprocessing
    preprocessor = SpotifyPipelinePreprocessor(target_col='popularity', test_size=0.2)
    X_train, X_test, y_train, y_test = preprocessor.process(df_clean)

    # Uruchomienie Tunera
    tuner = ModelTuner(n_iter=20, cv=3)  # Sprawdzi 20 losowych kombinacji
    best_params = tuner.tune(X_train, y_train)


    print("Optymalne parametry dla modelu XGBoost:")
    print(best_params)

    # Sprawdzenie na zbiorze testowym
    print("Trenowanie modelu z najlepszymi parametrami...")
    final_model = xgb.XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    final_model.fit(X_train, y_train)

    evaluator = ModelEvaluator()
    y_pred = final_model.predict(X_test)
    evaluator.evaluate(y_test, y_pred, model_name="Tuned XGBoost")


if __name__ == "__main__":
    run_tuning()

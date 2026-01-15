import logging
import os
import xgboost as xgb
from src.loaders import DataLoaderFactory
from src.cleaners import SpotifyDataCleaner
from src.preprocessors import SpotifyPipelinePreprocessor
from src.trainers import ModelTrainer
from src.evaluation import ModelEvaluator
from src.serializers import ModelSerializer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training_pipeline(version: str = 'v1'):
    logger.info("ROZPOCZYNANIE PROCESU TRENINGOWEGO")

    # Przygotowanie folderu na dane
    os.makedirs('data', exist_ok=True)

    # Ładowanie danych
    data_source = "hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv"
    loader = DataLoaderFactory.get_loader(data_source)
    raw_df = loader.load()
    raw_df.to_parquet(f'data/raw_data_{version}.parquet')

    # Czyszczenie danych
    cleaner = SpotifyDataCleaner()
    df_clean = cleaner.clean(raw_df)
    df_clean.to_parquet(f'data/clean_data_{version}.parquet')

    # Preprocessing-przygotowanie zbiorów danych do trenowania modelu
    preprocessor = SpotifyPipelinePreprocessor(target_col='popularity', test_size=0.2)
    X_train, X_test, y_train, y_test = preprocessor.process(df_clean)

    # Konfiguracja modelu
    logger.info("Inicjalizacja modelu XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )

    # Trening
    trainer = ModelTrainer()
    trainer.train(model, X_train, y_train)

    # Ewaluacja
    evaluator = ModelEvaluator()
    y_pred = model.predict(X_test)
    metrics = evaluator.evaluate(y_test, y_pred, model_name="Production XGBoost")


    # Serializacja
    serializer = ModelSerializer()

    # Serializacja modelu
    serializer.save(model, f'spotify-xgb-model_{version}.joblib')

    # Serializacja preprocessora
    serializer.save(preprocessor, f'spotify-preprocessor_{version}.joblib')

    logger.info("KONIEC PROCESU TRENINGOWEGO")


if __name__ == "__main__":
    run_training_pipeline()
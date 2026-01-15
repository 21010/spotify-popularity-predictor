import os
from src.serializers import ModelSerializer


def test_serializer_saves_and_loads(tmp_path):
    temp_dir = str(tmp_path)
    serializer = ModelSerializer(base_dir=temp_dir)

    # Przykładowe dane
    dummy_data = {"model_name": "test_xgboost", "accuracy": 0.99}
    filename = "test_model.joblib"

    # Zapis
    saved_path = serializer.save(dummy_data, filename)

    # Sprawdzanie, czy plik istnieje po zapisaniu
    assert os.path.exists(saved_path)

    # Odczyt
    loaded_data = serializer.load(filename)

    # Sprawdzanie, czy dane są takie same po odczycie
    assert loaded_data == dummy_data
    assert loaded_data['model_name'] == "test_xgboost"
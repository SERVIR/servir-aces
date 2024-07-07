import pytest
from aces.config import Config
from aces.model_trainer import ModelTrainer

@pytest.fixture(scope="module")
def train_model():
    config_file = "./tests/test_config.env"

    config = Config(config_file, override=True)
    print("Configuration loaded successfully.")

    print("Starting model training...")
    trainer = ModelTrainer(config)

    trainer.train_model()
    print("Model training completed.")

    trained_model_dir = config.MODEL_DIR / "trained-model"
    assert trained_model_dir.exists(), "Trained model directory not found."
    assert any(trained_model_dir.iterdir()), "Trained model files not found in the expected directory."

    return config

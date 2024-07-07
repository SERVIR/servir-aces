import pytest
from aces.config import Config

def test_model_config():
    config_file = "./tests/test_config.env"

    config = Config(config_file, override=True)
    print("Configuration loaded successfully.")

    # Add assertions to verify the configuration values
    assert str(config.DATADIR) == "tests/test_data", "Data directory not successfully configured in the configuration."

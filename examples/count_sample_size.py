# -*- coding: utf-8 -*-

try:
    from aces.config import Config
    from aces.data_processor import DataProcessor
except ModuleNotFoundError:
    print("ModuleNotFoundError: Attempting to import from parent directory.")
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from aces.config import Config
    from aces.data_processor import DataProcessor



if __name__ == "__main__":
    config_file = "config.env"
    config = Config(config_file)
    # expand the config
    additional_config = {
        "PRINT_DATASET": True
    }
    n_training_records, n_testing_records, n_validation_records = DataProcessor.calculate_n_samples(**{**config.__dict__, **additional_config})
    print(f"no of training records: {n_training_records}")
    print(f"no of testing records: {n_testing_records}")
    print(f"no of validation records: {n_validation_records}")

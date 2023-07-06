# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)

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
    config = Config()
    n_training_records, n_testing_records, n_validation_records = DataProcessor.calculate_n_samples(**config.__dict__)
    logging.info(f"no of training records: {n_training_records}")
    logging.info(f"no of testing records: {n_testing_records}")
    logging.info(f"no of validation records: {n_validation_records}")

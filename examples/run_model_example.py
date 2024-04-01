# -*- coding: utf-8 -*-
try:
    from aces.config import Config
    from aces.model_trainer import ModelTrainer
except ModuleNotFoundError:
    print("ModuleNotFoundError: Attempting to import from parent directory.")
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from aces.config import Config
    from aces.model_trainer import ModelTrainer



if __name__ == "__main__":
    config_file = "config.env"
    config = Config(config_file)
    trainer = ModelTrainer(config)
    trainer.train_model()

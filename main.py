# -*- coding: utf-8 -*-

import logging

from aces.config import Config
from aces.model_trainer import ModelTrainer



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = Config()
    trainer = ModelTrainer(config)
    trainer.train_model()

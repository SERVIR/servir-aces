# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from aces.config import Config
from aces.model_trainer import ModelTrainer



if __name__ == "__main__":
    config = Config()
    trainer = ModelTrainer(config)
    trainer.train_model()

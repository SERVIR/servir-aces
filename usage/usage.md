# Usage

To use servir-aces in a project:

```
from aces.config import Config
from aces.model_trainer import ModelTrainer

if __name__ == "__main__":
    config = Config()
    trainer = ModelTrainer(config)
    trainer.train_model()
```

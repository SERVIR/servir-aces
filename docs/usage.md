# Usage

To use servir-aces in a project.

```python
from aces.config import Config
from aces.model_trainer import ModelTrainer

if __name__ == "__main__":
    config_file = "config.env"
    config = Config(config_file)
    trainer = ModelTrainer(config)
    trainer.train_model()
```

Several notebooks are also available on the `notebook` folder implementing DL methods.

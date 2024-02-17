import json

import lightning as L
import torch
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from lightning.pytorch.tuner import Tuner

from cnn.datasets import IntelDataModule
from cnn.resnet_models import ResnetClassifier
from cnn.trainer import ModelTrainer

# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    with open("CNN/cnn/config.json" , "r") as config:
        config = json.load(config)
        print(config)

    data_config = config["Data_Config"]
    model_config = config["Model_Config"]

    data_module = IntelDataModule(data_dir=data_config["base_path"])

    classifier = ResnetClassifier(num_classes=model_config["num_classes"])
    model_trainer = ModelTrainer(classifier, learning_rate=model_config["learning_rate"],
                                 debug=True, use_discriminative_lr=False)

    trainer = L.Trainer(
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=model_config["patience"]),
            L.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        fast_dev_run=False
    )

    #     # Create a Tuner
    #     tuner = Tuner(trainer)

    #     tuner.lr_find(model_trainer, data_module)

    trainer.fit(model_trainer, data_module)
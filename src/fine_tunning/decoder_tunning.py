from .dataset import get_data_for_training
from .model import LitLMModel
from ..utility.settings import app_settings
from ..utility.utils import read_yaml
from ..utility.formater import alpaca_input_format
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.trainer import Trainer
from pathlib import Path
from typing import Dict, List, Union
from datasets import load_dataset
from transformers import AutoTokenizer


def auth_wandb() -> bool:
    status = wandb.login(key=app_settings.WANDB_API_KEY)
    return status


def init_trainer(training_config: Dict[str, Union[str, List]]) -> Trainer:

    assert auth_wandb() == True, "Failed to login wandb"

    # config wandbLogger
    logger = WandbLogger(**training_config["wandb"])
    logger.experiment.config.update(training_config["lora"])
    logger.experiment.config.update(training_config["model"])
    callbacks = [EarlyStopping(monitor="val_loss", mode="min")]
    trainer = Trainer(logger=logger, callbacks=callbacks, **training_config["trainer"])
    return trainer


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "config.yaml"
    training_config = read_yaml(config_path)
    trainer = init_trainer(training_config)

    ds = load_dataset(**training_config["dataset"])
    tokenizer = AutoTokenizer.from_pretrained(training_config["model"]["model_id"])
    train_data_config = {
        "pad_token_id": tokenizer.pad_token_id,
        "allowed_max_length": tokenizer.model_max_length,
        "mask_instruct": training_config["general"]["mask_instruct"],  # get from config
        "batch_size": training_config["general"]["batch_size"],  # get from config
        "num_workers": training_config["general"]["num_workers"],  # get from config
        "shuffle": True,
        "drop_last": True,
    }
    val_data_config = {
        "pad_token_id": tokenizer.pad_token_id,
        "allowed_max_length": tokenizer.model_max_length,
        "mask_instruct": training_config["general"]["mask_instruct"],  # get from config
        "batch_size": training_config["general"]["batch_size"],  # get from config
        "num_workers": training_config["general"]["num_workers"],  # get from config
        "shuffle": False,
        "drop_last": False,
    }
    trainer.logger.experiment.config.update({"train_data_config": train_data_config})
    trainer.logger.experiment.config.update({"val_data_config": val_data_config})
    train_loader = get_data_for_training(
        ds=ds["train"],
        tokenizer=tokenizer,
        formater=alpaca_input_format,
        config=train_data_config,
    )
    val_loader = get_data_for_training(
        ds=ds["validation"],
        tokenizer=tokenizer,
        formater=alpaca_input_format,
        config=val_data_config,
    )
    model = LitLMModel(
        model_path=training_config["model"]["model_id"],
        lora_config=training_config["lora"],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

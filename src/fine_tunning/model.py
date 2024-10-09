from transformers import AutoModelForCausalLM
import lightning as L
from peft import LoraConfig, get_peft_model
import torch
from typing import Dict,Union,List

class LitLMModel(L.LightningModule):
    def __init__(self, model_path: str, lora_config: Dict[str,Union[str,List[str]]]):
        super(LitLMModel, self).__init__()
        # save hyperparamters
        self.save_hyperparameters()
        # Load base model from HF
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=self.device, trust_remote_code=True
        )
        # Freeze all paramters
        self._freeze_all_parameters(model)
        # create config for Lora 
        self.config = LoraConfig(**lora_config)
        # apply optimization from PEFT
        self.model = get_peft_model(model, self.config)
        # Loss function
        self.metric = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx, "train")
        return loss

    def _freeze_all_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx, "test")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch)

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        pred = self.model(x.to(self.device))
        loss = self.metric(pred.logits.flatten(0, 1), y.flatten())
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=4.21e-05, weight_decay=0.1)
        return optimizer

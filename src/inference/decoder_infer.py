from ..fine_tunning.model import LitLMModel
from ..fine_tunning.dataset import get_data_for_inference
from ..utility.formater import alpaca_input_format
from ..fine_tunning.decoder_tunning import auth_wandb
from ..utility.utils import read_yaml
from pathlib import Path
import wandb
def download_checkpoint(chekpoint_path:str):
    auth_wandb()
    run = wandb.init()
    artifact = run.use_artifact(chekpoint_path, type='model')
    artifact_dir = artifact.download()
    run.finish()
    return Path(artifact_dir)/"model.ckpt"


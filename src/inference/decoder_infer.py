from typing import Dict, List, Union
from pathlib import Path
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from ..fine_tunning.model import LitLMModel
from ..fine_tunning.dataset import get_data_for_inference
from ..utility.formater import alpaca_input_format
from ..fine_tunning.decoder_tunning import auth_wandb
from ..utility.utils import read_yaml


def download_checkpoint(checkpoint_path: str) -> Path:
    auth_wandb()
    with wandb.init() as run:
        artifact = run.use_artifact(checkpoint_path, type="model")
        artifact_dir = artifact.download()
    return Path(artifact_dir) / "model.ckpt"


def get_loader(
    config: Dict[str, Union[str, List[str]]],
    model_tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
):
    test_ds = load_dataset(config["dataset"], split="test")
    test_data_config = {
        "pad_token_id": model_tokenizer.pad_token_id,
        "allowed_max_length": model_tokenizer.model_max_length,
        "mask_instruct": config["general"]["mask_instruct"],  # get from config
        "batch_size": config["general"]["batch_size"],  # get from config
        "num_workers": config["general"]["num_workers"],  # get from config
        "shuffle": False,
        "drop_last": False,
    }
    test_data_loader = get_data_for_inference(
        test_ds,
        tokenizer=model_tokenizer,
        formater=alpaca_input_format,
        config=test_data_config,
    )
    return test_data_loader


def do_inference(
    model: LitLMModel,
    config: Dict[str, str],
    test_data_loader: DataLoader,
    model_tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
) -> pd.DataFrame:
    reference_summary = []
    pred_summary = []
    for batch in tqdm(test_data_loader):
        generate_ids = model.predict(batch["input_ids"], config)
        response = model_tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
        )
        pred_summary.extend(response)
        reference_summary.extend(batch["responses"])

    results = pd.DataFrame({"prediction": pred_summary, "reference": reference_summary})
    results.loc[:, "prediction"] = results["prediction"].apply(
        lambda x: x.split("### الرد المناسب:\n")[-1]
    )
    return results


if __name__ == "__main__":
    infer_config_path = Path(__file__).parent / "config.yaml"
    tunning_config_path = Path(__file__).parent.parent / "fine_tunning" / "config.yaml"
    infer_config = read_yaml(infer_config_path)
    tunning_config = read_yaml(tunning_config_path)
    tokenizer = AutoTokenizer.from_pretrained(tunning_config["model"]["model_id"])
    test_loader = get_loader(config=tunning_config, model_tokenizer=tokenizer)
    checkpoint = download_checkpoint(
        checkpoint_path=infer_config["inference"]["checkpoint"]
    )
    litmodel = LitLMModel.load_from_checkpoint(
        checkpoint,
        model_path=tunning_config["model"]["model_id"],
        lora_config=tunning_config["lora"],
    )
    litmodel.eval()

    df = do_inference(
        model=litmodel,
        test_data_loader=test_loader,
        config=infer_config["generation"],
        model_tokenizer=tokenizer,
    )

    df.to_csv(infer_config["inference"]["prediction_path"], index=False)

from typing import (Dict,
                     Any,
                     Callable)
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer


class CustomInstructDataset(Dataset):
    def __init__(
        self, data: HFDataset, tokenizer: AutoTokenizer, formater: Callable[[str], str]
    ):
        super(CustomInstructDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.formater = formater

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        model_instruction = self.formater(entry)
        desired_response = f"\n\n### الرد المناسب:\n{entry['response']}"
        return (
            self.tokenizer.encode(model_instruction),
            self.tokenizer.encode(desired_response),
        )


def custom_callate(
    batch,
    ignore_index=-100,
    pad_token_id=50256,
    allowed_max_length=128,
    mask_instruct=False,
):
    max_batch_len = max([len(entry[0]) + len(entry[1]) + 1 for entry in batch])
    inputs = []
    targets = []
    for instruct, response in batch:

        padding = [pad_token_id] * (max_batch_len - (len(instruct) + len(response)))

        full_seq = instruct + response + padding
        input_pt = torch.tensor(full_seq)[:-1]
        if mask_instruct:
            masked_instruct = [ignore_index] * len(instruct[1:])
            masked_instr_seq = masked_instruct + response + padding

            target_pt = torch.tensor(masked_instr_seq)
        else:
            target_pt = torch.tensor(full_seq[1:])

        mask = target_pt == pad_token_id
        indices_to_mask = torch.nonzero(mask).squeeze()
        if indices_to_mask.numel() > 1:
            indices_skiped_first = indices_to_mask[1:]
            target_pt[indices_skiped_first] = ignore_index

        input_pt = input_pt[:allowed_max_length]
        target_pt = target_pt[:allowed_max_length]

        inputs.append(input_pt)
        targets.append(target_pt)
    res_inputs = torch.stack(inputs)
    res_targets = torch.stack(targets)
    return res_inputs, res_targets


def get_data_for_training(
    ds: HFDataset,
    tokenizer: AutoTokenizer,
    formater: Callable[[str], str],
    config: Dict[str, Any],
) -> DataLoader:
    """
    Prepares a DataLoader for training a model using the provided dataset, tokenizer,
    formatting function, and configuration settings.

    Args:
        ds (HFDataset): The Hugging Face dataset containing the training data.
        tokenizer (AutoTokenizer): The tokenizer used to process the input text.
        formater (Callable[[str], str]): A function that formats the text input as needed.
        config (Dict[str, Any]): A dictionary containing configuration settings such as:
            - 'pad_token_id'[int]: Token ID used for padding.
            - 'allowed_max_length'[int]: Maximum length allowed for input sequences.
            - 'mask_instruct'[bool]: Do Instruction masking by default = `False`.
            - 'batch_size'[int]: Size of the batches for the DataLoader.
            - 'shuffle'[bool]: Boolean indicating whether to shuffle the dataset.
            - 'drop_last'[bool]: Boolean indicating whether to drop the last incomplete batch.
            - 'num_workers'[int]: Number of worker processes for data loading.

    Returns:
        DataLoader: A DataLoader instance that yields batches of formatted input data
        ready for model training.
    
    Note:
        This function sets a manual seed for reproducibility and uses a customized
        collate function to handle batch formation.
    """
    torch.manual_seed(123)
    customized_collate_fn = partial(
        custom_callate,
        pad_token_id=config["pad_token_id"],
        allowed_max_length=config["allowed_max_length"],
        mask_instruct=config["mask_instruct"],
    )
    custom_ds = CustomInstructDataset(ds["train"], tokenizer, formater)
    ds_loader = DataLoader(
        custom_ds,
        batch_size=config["batch_size"],
        collate_fn=customized_collate_fn,
        shuffle=config["shuffle"],
        drop_last=config["drop_last"],
        num_workers=config["num_workers"],
    )
    return ds_loader

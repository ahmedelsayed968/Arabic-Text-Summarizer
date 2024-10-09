from typing import Dict, Any, Callable, List, Tuple
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer


class CustomInstructDataset(Dataset):
    def __init__(
        self, data: HFDataset, tokenizer: AutoTokenizer, formater: Callable[[str], str]
    ):
        super().__init__()
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


class CustomInstructInferenceDataset(Dataset):
    """
    A custom dataset class for inference, designed to provide model instructions
    and their corresponding responses from a Hugging Face dataset.

    Args:
        data (HFDataset): A Hugging Face dataset containing the data entries,
                          each entry should have an 'response' field.
        tokenizer (AutoTokenizer): The tokenizer used for processing input text.
        formater (Callable[[str], str]): A function that formats the input text
                                          to the desired instruction format.

    Methods:
        __len__() -> int:
            Returns the total number of entries in the dataset.

        __getitem__(index: int) -> Tuple[str, str]:
            Retrieves the formatted instruction and the corresponding response
            for the specified index.
    """

    def __init__(
        self, data: HFDataset, tokenizer: AutoTokenizer, formater: Callable[[str], str]
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.formater = formater

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        model_instruction = self.formater(entry)
        return (model_instruction, entry["response"])


def custom_callate_inference(batch: List[Tuple[str, str]], tokenizer: AutoTokenizer):
    """
    Custom collate function for processing a batch of model instructions and
    their corresponding responses for inference.

    Args:
        batch (list): A list of tuples, where each tuple contains a model
                      instruction (str) and a response (str).
        tokenizer (AutoTokenizer): The tokenizer used to convert model instructions
                                   into input IDs suitable for the model.

    Returns:
        dict: A dictionary containing:
            - 'input_ids' (torch.Tensor): A tensor of tokenized model instructions,
              with padding and truncation applied.
            - 'responses' (list): A list of responses corresponding to the model
              instructions, preserving their original order.

    Note:
        This function ensures that model instructions are tokenized and padded to
        the same length, making them ready for input into a model during inference.
    """
    # Separate model instructions and responses
    model_instructions, responses = zip(*batch)

    # Tokenize model instructions
    tokenized_inputs = tokenizer(
        list(model_instructions),  # Ensure inputs are a list
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Return the tokenized inputs and responses
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "responses": list(responses),  # Ensure responses are a list
    }


def custom_callate(
    batch,
    ignore_index=-100,
    pad_token_id=50256,
    allowed_max_length=128,
    mask_instruct=False,
):
    max_batch_len = max(len(entry[0]) + len(entry[1]) + 1 for entry in batch)
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


def get_data_for_inference(
    ds: HFDataset,
    tokenizer: AutoTokenizer,
    formater: Callable[[str], str],
    config: Dict[str, Any],
) -> DataLoader:
    """
    Prepares a DataLoader for inference from a given Hugging Face dataset.

    Args:
        ds (HFDataset): The dataset containing the test data to be used for inference.
        tokenizer (AutoTokenizer): The tokenizer used to convert model instructions
                                   into input IDs.
        formater (Callable[[str], str]): A function that formats the input text to
                                          the desired instruction format.
        config (Dict[str, Any]): A configuration dictionary containing parameters such as
                                  'batch_size' and 'num_workers'.

    Returns:
        DataLoader: A DataLoader instance configured for the inference dataset,
                     with batched inputs and collated outputs.

    Notes:
        - The function sets a manual seed for reproducibility.
        - It uses a custom collate function to handle the input processing for the model.
        - The DataLoader is configured to not shuffle the dataset and does not drop the last
          batch, ensuring all data points are used during inference.
    """
    torch.manual_seed(123)
    custom_collate_fn = partial(custom_callate_inference, tokenizer=tokenizer)
    custom_ds = CustomInstructDataset(ds["test"], tokenizer, formater=formater)
    ds_loader = DataLoader(
        custom_ds,
        batch_size=config["batch_size"],
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=config["num_workers"],
    )
    return ds_loader

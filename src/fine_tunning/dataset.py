from torch.utils.data import Dataset,DataLoader
from typing import Callable
from datasets import Dataset as HFDataset
import torch
from transformers import AutoTokenizer
class CustomInstructDataset(Dataset):
    def __init__(self,
                 data:HFDataset,
                 tokenizer:AutoTokenizer,
                 formater:Callable[[str],str]):
        super(CustomInstructDataset,self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.formater = formater
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        model_instruction = self.formater(entry)
        desired_response = f"\n\n### الرد المناسب:\n{entry['response']}"
        return (self.tokenizer.encode(model_instruction),
                self.tokenizer.encode(desired_response))
    

def custom_callate(batch,
                   device="cpu",
                   ignore_index=-100,
                   pad_token_id=50256,
                   allowed_max_length=128,
                   mask_instruct=False):
    max_batch_len = max([len(entry[0])+len(entry[1])+1 for entry in batch])
    inputs = []
    targets = []
    for instruct,response in batch:

        padding = [pad_token_id]*(max_batch_len-(len(instruct)+len(response)))

        full_seq = instruct+response+padding
        input_pt = torch.tensor(full_seq)[:-1]
        if mask_instruct:
            masked_instruct = [ignore_index]*len(instruct[1:])
            masked_instr_seq = masked_instruct+response+padding

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
    res_inputs = torch.stack(inputs).to(device)
    res_targets = torch.stack(targets).to(device)
    return res_inputs,res_targets


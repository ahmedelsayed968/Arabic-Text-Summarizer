from torch.utils.data import Dataset,DataLoader
from typing import Callable
from datasets import Dataset as HFDataset
# from tokenizers import Tokenizer
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
    


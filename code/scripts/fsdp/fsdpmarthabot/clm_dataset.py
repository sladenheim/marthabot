import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

class marthabot_clm_dataset(Dataset):
    def __init__(self, dataset_path):
        """
        This is a generic PyTorch dataset for causal language modeling for marthabot/other style bots.
        Assumes all necessary pre-processing is done - just loading in the cleaned hugging face data. 
        Assumes each example has 'input_ids', 'attention_mask', and 'labels'.
        """
        self.dataset = load_from_disk(dataset_path)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Convert Hugging Face dataset example to PyTorch tensors - this is what data loader expects
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)

        # return as dict
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

#from clm_dataset import marthabot_clm_dataset

#def get_dataset(path):
    #return CLMDataset(path)


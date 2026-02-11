from dataclasses import dataclass

# Contains paths to cleaned and pre-processed datasets for training and testing. 
@dataclass
class data_config:
    train_dataset_path: str = "/projectnb/scottml/seansal2/data/datasets/blood_memory_clm_train_new"
    test_dataset_path: str = "/projectnb/scottml/seansal2/data/datasets/blood_memory_clm_test_new"

# Update to use new data with new tokenizer (llama 3 generation)
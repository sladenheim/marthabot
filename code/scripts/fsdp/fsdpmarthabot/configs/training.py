from dataclasses import dataclass
from typing import ClassVar

# Contains important pytorch training configuration settings.
@dataclass
class train_config:
    # Adjust model name here 
    model_name: str="meta-llama/Llama-3.1-8B" # formerly "meta-llama/Llama-2-7b-hf" and "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-8B"
    run_validation: bool=True
    batch_size_training: int=4
    num_workers_dataloader: int=2
    lr: float=5e-6 # should probably decrease # default was 0.002
    weight_decay: float=0.1 # should probably actually use for L2 regularization! # default was 0.0
    gamma: float= 0.95
    use_fp16: bool=False
    mixed_precision: bool=True
    save_model: bool=True

    
    
    
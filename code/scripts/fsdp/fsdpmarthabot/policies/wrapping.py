# holds various wrapping policies for fsdp


import torch.distributed as dist
import torch.nn as nn
import torch

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

# If we want to do exact wrapping, must import the corresponding model layer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import functools
from typing import Type


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


def get_specific_llm_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    NOTE: for now, have it set to llama - get add it to a config file 
    """
    # ====   use new transformer wrapper

    llm_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer}, #This can be whatever layer we need
    )
    return llm_auto_wrap_policy

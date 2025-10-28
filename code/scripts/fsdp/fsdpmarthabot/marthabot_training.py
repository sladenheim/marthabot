import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#NEW: Change from t5-specific  to HuggingFace AutoClass
from transformers import AutoTokenizer, AutoModelForCausalLM
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
# NEW: import marthabot_clm_dataset
from clm_dataset import marthabot_clm_dataset
import policies
import model_checkpointing
from configs import fsdp_config, train_config, data_config
from utils import (bfloat_support, setup,
                   cleanup, get_date_of_run,
                   format_metrics_to_gb,
                   train,validation,setup_model)
# Removed t5 modeling and t5block imports - import auto wrap code
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools
# NOTE: Can load in model specific layers so we can utilize the auto_wrap function
# For Llama2:
from transformers.models.llama.modeling_llama import LlamaDecoderLayer 
from typing import Type
import time
import tqdm
from datetime import datetime
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from packaging.version import Version, parse
from policies import bfSixteen


def get_policies(cfg, rank):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if rank == 0:
                print(f"FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    wrapping_policy = policies.get_specific_llm_wrapper() #see policies for exact info

    return mixed_precision_policy, wrapping_policy


def fsdp_main(args):

    # Pull environment vars --> use them to manage distributed training
    local_rank = int(os.environ['LOCAL_RANK']) # dictate which GPU to use
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if local_rank == 0:
        cache_dir = os.path.join(os.environ.get("TMPDIR"), "martha_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    # Load in Automodel and AutoTokenizer - see setup_model() function for specifics/more customization

    model, tokenizer = setup_model(train_config.model_name) #this might be bad practice - look more into it

    # Load datasets from file paths in the config
    train_dataset = marthabot_clm_dataset(data_config.train_dataset_path)
    val_dataset = marthabot_clm_dataset(data_config.test_dataset_path)

    # Ensure each process gets a unique shard/part of the dataset
    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    # Start the process group 
    setup()

    # Establish arguemnts/configs for DataLoader, dictates how PyTorch feeds data into model
    # Set up batch size, makes sure each GPU sees different data
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    # set cuda settings: num_workers = number of cpu threads used to load batches, pin_memory = allocates data in page-locked memory, makes GPU transfers faster, don't need shuffling since DistributedSampler does it
    cuda_kwargs = {'num_workers': 2, 
                    'pin_memory': True,
                    'shuffle': False}
    # combine everything 
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    # Create the dataloaders, pass in the arguments set above
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    # IMPORTANT feature: Set auto wrap policy. Takes in a layer, decides to wrap/not.
        # we only pick certain layers to wrap (ones with a lot of parameters) --> these layers are the ones sharded/split between GPUs/processes
            # split layer parameters, gather them when needed, free memory 
        # transformer_auto_wrap_policy - special prebuilt function for sharding with Transformers 
    # In this case, only wrap layers that are instances of T5Block - basic transformer block of the HF T5 model
    

    # IMPORTANT: Sharding strategy - choose how to split model
        # Default: fully shard model parameters, gradients, optimizer states acros all ranks = Zero3 
        # _GRAD_OP = Zero2 = only optimizer states and gradients sharded --> this reduces communication overhead in FSDP. Saves an all_Gather during backwards pass
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD #for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank) #for this process, use local_rank as default CUDA device


    #init_start_event = torch.cuda.Event(enable_timing=True)
    #init_end_event = torch.cuda.Event(enable_timing=True)

    #init_start_event.record()

    bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and parse(torch.version.cuda) >= parse("11.0")
    and dist.is_nccl_available()
    and torch.cuda.nccl.version() >= (2, 10)
    )

    # Check if BF16 precision is supported, very fast mixed-precision format
    if bf16_ready:
        mp_policy = bfSixteen
    else:
        mp_policy = None # defaults to fp32

    # Set up FSDP parameters
    mp_policy, auto_wrapping_policy = get_policies(train_config, rank)
    
    # model is on CPU before input to FSDP
    # This wraps the model in FSDP (according to our policy), with mixed precision, and sets the right GPU id
    model = FSDP(model,
        auto_wrap_policy=auto_wrapping_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())
    # Set up optimizer 
    optimizer = optim.SGD(model.parameters(), lr=train_config.lr)
    # StepLR decays learning rate each epoch by gamma
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    # Can adjust the file save name to match the model
    file_save_name = f"{train_config.model_name}-Save"

    # Only rank 0/first GPU tracks time, accuracy etc --> is the main process
    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    # Training loop 
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        if train_config.run_validation: # checks validation accuracy if desired, adjusts learning rate accordingly
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()

        # Logs epoch time, accuracy/loss, memory usage
        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if train_config.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print(f"completed save and stats zone...")
            
        # Save only the best models
        # Save only the best models
        if train_config.save_model and curr_val_loss < best_val_loss:

            if rank == 0:
                print(f"--> entering save model state")

            if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                model_checkpointing.save_model_checkpoint(
                    model, optimizer, rank, fsdp_config, epoch=epoch
                )

            elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                model_checkpointing.save_model_and_optimizer_sharded(
                    model, rank, fsdp_config
                )
                if fsdp_config.save_optimizer:
                    model_checkpointing.save_model_and_optimizer_sharded(
                        model, rank, fsdp_config, optim=optimizer
                    )

            if fsdp_config.save_optimizer:
                model_checkpointing.save_optimizer_checkpoint(
                    model, optimizer, rank, fsdp_config, epoch=epoch
                )


        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()



if __name__ == '__main__':
    # Training settings - below specified in command line
    # Adjust description here
    parser = argparse.ArgumentParser(description='Marthabot FSDP')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    os.environ.get("TMPDIR")
    tmp_dir = os.environ.get("TMPDIR")
    cache_dir = os.path.join(tmp_dir, "martha_cache")

    # Do we need to spawn anything here? Add as a test
    fsdp_main(args)

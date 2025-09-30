# Why FSDP? - We need model paralleism since our model does not fit on our GPU.
# How does it work? - Shard/split model parameters, optimizer states, and gradients across DDP ranks 
# What are the stages/steps?
    # Constructor - 
        # Shard model parameters and each rank only keeps its own shard
    # Forward Pass - 
        # Run all_gather to collect all shards from all ranks to recover the full parameter for this FSDP unit Run forward computation
        # Discard non-owned parameter shards it has just collected to free memory
    # Backward Pass
        # Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit Run backward computation
        # Discard non-owned parameters to free memory.
        # Run reduce_scatter to sync gradients
# Important functions
    # auto_wrap_policy() - feature that ...
    # all_gather - 
    # reduce_scatter - 



# TOY EXAMPLE: Fine-Tuning T5 Model for text summarization. Single node. 8 A100 GPUs. 

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.t5.modeling_t5 import T5Block
from nlp import load_dataset

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
from summarization_dataset import *
import policies
import model_checkpointing
from configs import fsdp_config, train_config
from utils import (bfloat_support, setup,
                   cleanup, get_date_of_run,
                   format_metrics_to_gb,
                   train,validation,setup_model)
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
import tqdm
from datetime import datetime
import os
import torch
import torch.distributed as dist
from datetime import datetime
import tqdm
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration

g_gigabyte = 1024**3

#################################
# PART 1: Setup Helper functions, training, validation
#################################
# First setup helper functions...
    # Model parallelism requires communication b/w multiple GPUs/processes
    # This communication is handle by a process group
def setup():
    # initialize the process group - so GPUs can talk to each other
    # nccl - backend used for GPU communication on NVIDIA hardware
    # once group is established - 
        # each process knows total workers (WORLD_SIZE)
        # its own id/rank
        # how to send/receive data to and from others
    # torchrun sets worker RANK and WORLD_SIZE automatically
    dist.init_process_group("nccl")

def cleanup():
    # shuts down communication between processes - free up resources
    dist.destroy_process_group()

# Set up the model
def setup_model(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer =  T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Helper functions for data + formatting memory 
def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


##############################################
# PART 2: Set up train and validation function
##############################################

# Define a train function: 
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    # args = general config/settings, model = FSDP-wrapped model
    # rank = process/gpu ID, world_size = total number of processes/GPUs
    # train_loader = loads in training data in batches 
    # optimizer = pick what optimizer to use for backprop and parameter updates
    # sampler - shuffles dataset each epoch 
    
    model.train() # put model in training mode
    local_rank = int(os.environ['LOCAL_RANK']) #tells script which GPU on current machine this process should use ie.) Rank = 0, local_rank = 0 --> use GPU 0
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler: # set different sampling for each epoch if desired
        sampler.set_epoch(epoch)
    if rank==0: # progress bar 
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    # Main training loop - forward/backward pass
    for batch in train_loader: #iterate over the batches of data
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank) #move each tensor (input_id, mask, target_id) to the correct GPU
        optimizer.zero_grad() #clear old gradients from last batch before calculating new ones
        # next do the forward pass --> pass input data into model (embedding lookup, pass through dense + transformer layers, output prediction, calculate loss)
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        loss = output["loss"]
        loss.backward() #compute gradients of loss with respect to all model parameters
        optimizer.step() #use gradients to update model weights
        fsdp_loss[0] += loss.item() # add to total loss
        fsdp_loss[1] += len(batch) # add to sample count
        if rank==0: #update progress bar only for 0th gpu
            inner_pbar.update(1)

    # Combine/reduce all local loss counts across all GPUs into single total
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]

    # Print loss
    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy

# Very similar for validation - no weight updates/gradients, just evaluation
def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss
##############################################
# PART 3: Wrap model in FSDP 
##############################################

# Set up wikihow() function
# Set up DistriubtedSampler() 
# Are these predefined or do we write them?

def fsdp_main(args):

    # Load model, 
    model, tokenizer = setup_model("t5-base")

    # Pull environment vars --> use them to manage distributed training
    local_rank = int(os.environ['LOCAL_RANK']) # dictate which GPU to use
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Load dataset
    dataset = load_dataset('wikihow', 'all', data_dir='data/')
    print(dataset.keys())
    print("Size of train dataset: ", dataset['train'].shape)
    print("Size of Validation dataset: ", dataset['validation'].shape)


    #wikihow(tokenizer, type_path, num_samples, input_length, output_length, print_text=False)
    # tokenize 
    train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False)
    val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False)

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
    
    t5_auto_wrap_policy = functools.partial( 
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )

    # IMPORTANT: Sharding strategy - choose how to split model
        # Default: fully shard model parameters, gradients, optimizer states acros all ranks = Zero3 
        # _GRAD_OP = Zero2 = only optimizer states and gradients sharded --> this reduces communication overhead in FSDP. Saves an all_Gather during backwards pass
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank) #for this process, use local_rank as default CUDA device


    #init_start_event = torch.cuda.Event(enable_timing=True)
    #init_end_event = torch.cuda.Event(enable_timing=True)

    #init_start_event.record()

    bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and LooseVersion(torch.version.cuda) >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )

    # Check if BF16 precision is supported, very fast mixed-precision format
    if bf16_ready:
        mp_policy = bfSixteen
    else:
        mp_policy = None # defaults to fp32

    # model is on CPU before input to FSDP
    # This wraps the model in FSDP (according to our policy), with mixed precision, and sets the right GPU id
    model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mp_policy,
        #sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())
    # Set up optimizer 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # StepLR decays learning rate each epoch by gamma
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "T5-model-"

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
        if args.run_validation: # checks validation accuracy if desired, adjusts learning rate accordingly
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()

        # Logs epoch time, accuracy/loss, memory usage
        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.run_validation:
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
        if args.save_model and curr_val_loss < best_val_loss:

            # save
            if rank == 0:
                print(f"--> entering save model state")

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            #print(f"saving process: rank {rank}  done w state_dict")


            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                print(f"--> attempting to save model prefix {currEpoch}")
                save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                print(f"--> saving as model name {save_name}")

                torch.save(cpu_state, save_name)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()

    
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

    wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy, wrapping_policy






 
    
##############################################
# PART 4: Define main function 
##############################################


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
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

    fsdp_main(args)
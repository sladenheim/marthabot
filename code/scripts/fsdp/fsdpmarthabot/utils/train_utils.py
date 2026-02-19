import os
import torch
import torch.distributed as dist
from datetime import datetime
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast #not sure what the GPT2Tokenizer was for

g_gigabyte = 1024**3

# OLD SETUP 
# def setup():
#     # initialize the process group
#     dist.init_process_group("nccl")

# NEW SETUP - ensure each rank uses its own GPU in setup()
def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

# OLD CLEANUP FUNCTION
def cleanup():
    dist.destroy_process_group()

# NEW/EDITED CLEANUP: 
# def cleanup():
#     # drain CUDA work on each rank
#     try:
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#     except Exception as e:
#         print(f"[cleanup] cuda sync warn: {e}")

#     # line up all ranks, then destroy
#     if dist.is_available() and dist.is_initialized():
#         try:
#             dist.barrier()
#             dist.destroy_process_group()
#         except Exception as e:
#             print(f"[cleanup] destroy warn: {e}")


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

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"] )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy


def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])
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


def setup_model(model_name):
    # QUESTION: Should we specify any other settings here for optimized training?
    """
    Loads a causal language model and tokenizer.
    Assumes model is decoder-only (e.g. LLaMA, GPT-2, Mistral).
    Automatically sets pad_token if needed.
    """
    
    cache_dir = os.path.join(os.environ.get("TMPDIR"), "martha_cache")    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Ensure pad_token exists (required for LLaMA-style models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Should we add data collator here?
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16  # or torch.float16 if you're using mixed precision
    )

    return model, tokenizer

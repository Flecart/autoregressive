import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import SimpleDecoderTransformer
from .dataset import MathsDataset
from time import time
import os
from .dataset import Tokenizer
from . import karpathy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR
import math
torch.set_float32_matmul_precision('high')

# Initialize wandb
use_wandb = True

# Model parameters
tokenizer = Tokenizer()
vocab_size = tokenizer.vocab_size
block_size = 512
n_embd = 128
n_head = 8
n_layer = 6
config = karpathy.GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)

# Training parameters
num_epochs = 50
batch_size = 1024
# steps = 990_000 / 2048 * 600 = ~293750
lr_decay_iters = 800_000
num_workers = 20
gradient_accumulation_steps = 4 # not used
warmup_steps = 1
learning_rate =  0.001 # 6e-4 # max learning rate
min_lr = 6e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Instantiate the model
# model = SimpleDecoderTransformer(vocab_size, block_size, n_embd, n_head, n_layer)
model: karpathy.GPT = karpathy.GPT(config)
weights = torch.load("llms/out/old/model_56000.pt")
unwanted_prefix = '_orig_mod.'
for k,v in list(weights.items()):
    if k.startswith(unwanted_prefix):
        weights[k[len(unwanted_prefix):]] = weights.pop(k)
model.load_state_dict(weights)

# model = torch.compile(model)
# If multiple GPUs are available, wrap model with nn.DataParallel
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model, device_ids=[0, 1], output_device=0, dim=0)
# Print number of parameters of th emodel
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Define a lambda function for the warmup
def lr_lambda(it: int):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return learning_rate * it / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (lr_decay_iters - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss(reduction='none')

# When creating the DataLoader, pass the collate_fn
train_dataset = MathsDataset("train")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
val_dataset = MathsDataset("val")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

steps = 0

# set up distributed training
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    print("Running in DDP mode")
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    model = model.to(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    print("Running in single GPU/CPU mode")
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

    if torch.cuda.is_available():
        device= 'cuda:0'
        model: SimpleDecoderTransformer = model.to(device)
    else:
        device = "cpu"
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), 'cuda')
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

raw_model = model
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

if use_wandb and master_process:
    wandb.init(project='owt', name='small-gpt-maths-v2')
# TODO list:
# - make evaluation on val in the training loop (this is the downstream task)
# - add perplexity
# - train and train.

def evaluate_downstream(model, batch, device):
    input, targets, mask = batch
    input = input.to(device)
    targets = targets.to(device)
    mask = mask.to(device)
    
    with torch.no_grad():
        output = model(input)
    

# Instantiate the scheduler
scheduler = LambdaLR(optimizer, lr_lambda)

# add tqdm for steps in epoch
pbar = tqdm(total=num_epochs*len(train_dataloader))
best_loss = 10
first_debug_print = True
# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Get input and target sequences from batch
        input_seq, target_seq, mask = batch

        # Debug prints here
        # if first_debug_print:
        #     numpy_input = input_seq.numpy()
        #     print(tokenizer.decode(numpy_input[0]))
        #     mask_input = mask.numpy()
        #     for x in mask_input[0]:
        #         print(x, end="")
        #     print()
        #     numpy_target = target_seq.numpy()
        #     print(tokenizer.decode(numpy_target[0]))
        #     print(input_seq.size())

        #     first_debug_print = False
        # asdfasdf

        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        mask = mask.to(device)
        # Forward pass
        output = model(input_seq)

        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Compute loss
        loss_all = loss_fn(output.view(-1, vocab_size), target_seq.view(-1)) * mask.view(-1)
        # print(loss_all)
        loss = loss_all.mean()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if steps % 500 == 0 and master_process:
                model.eval()
                perplexity_train = raw_model.get_perplexity(output, target_seq)
                for val_batch in val_dataloader:
                    inputs, targets, mask = val_batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    mask = mask.to(device)
                    outputs = model(inputs)
                    loss_all = loss_fn(output.view(-1, vocab_size), targets.view(-1)) * mask.view(-1)
                    val_loss = loss_all.mean()
                    break # only do one batch for now
                perplexity = raw_model.get_perplexity(output, targets)
                model.train()
        debug_learning_rate = lr_lambda(steps)
        if use_wandb and master_process:
            info = {"loss": loss.item(), 
                    "val_loss": val_loss.item(), 
                    "learning_rate": debug_learning_rate, 
                    "perplexity": perplexity,
                    "perplexity_train": perplexity_train
                }
            wandb.log(info)

        steps += 1
        if steps % 100 == 0 and master_process:
            print(f'Step [{steps}], Loss: {loss.item()}, Val_loss: {val_loss.item()}, LR: {debug_learning_rate}, Perplexity: {perplexity}, Perplexity_train: {perplexity_train}')
            if epoch == 0 or loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'llms/out/best_model.pt')

        if steps % 8000 == 0 and master_process:
            torch.save(model.state_dict(), f'llms/out/model_{steps}.pt')

        pbar.set_description(f'Loss: {loss.item()}, LR: {debug_learning_rate}')
        pbar.update(1)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


if use_wandb and master_process:
    wandb.finish()

if ddp:
    destroy_process_group()


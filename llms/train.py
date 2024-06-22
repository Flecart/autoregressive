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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Initialize wandb
use_wandb = False
if use_wandb:
    wandb.init(project='owt', name='small-gpt-maths-v2')

# Model parameters
vocab_size = 14
block_size = 512
n_embd = 128
n_head = 8
n_layer = 6

# Training parameters
num_epochs = 10
batch_size = 512
num_workers = 20
gradient_accumulation_steps = 4

# Instantiate the model
model = SimpleDecoderTransformer(vocab_size, block_size, n_embd, n_head, n_layer)
model = torch.compile(model)
model = model.to('cuda:0')
# If multiple GPUs are available, wrap model with nn.DataParallel
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model, device_ids=[0, 1], output_device=0, dim=0)
# Print number of parameters of th emodel
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters())

# When creating the DataLoader, pass the collate_fn
train_dataset = MathsDataset("llms/data/train.bin")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

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

# add tqdm for steps in epoch
pbar = tqdm(total=num_epochs*len(train_dataloader))

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Get input and target sequences from batch
        input_seq, target_seq, mask = batch

        # Debug prints here
        # numpy_input = input_seq.numpy()
        # tokenizer = Tokenizer()
        # print(tokenizer.decode(numpy_input[0]))
        # mask_input = mask.numpy()
        # for x in mask_input[0]:
        #     print(x, end="")
        # print()
        # numpy_target = target_seq.numpy()
        # print(tokenizer.decode(numpy_target[0]))
        # asdfasdf

        input_seq = input_seq.to('cuda:0')
        target_seq = target_seq.to('cuda:0')
        mask = mask.to('cuda:0')
        # Forward pass
        output = model(input_seq)

        # Compute loss
        
        loss_all = loss_fn(output.view(-1, vocab_size), target_seq.view(-1)) * mask.view(-1)
        loss = loss_all.mean()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics to wandb
        if use_wandb:
            wandb.log({"loss": loss.item()})

        # Print loss every 100 steps
        steps += 1

        if steps % 100 == 0:
            print(f'Step [{steps}], Loss: {loss.item()}')
            # Keep track of best model
            if epoch == 0 or loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'llms/out/best_model.pt')

        if steps % 2000 == 0:
            torch.save(model.state_dict(), f'llms/out/model_{steps}.pt')

        pbar.update(1)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


if use_wandb:
    wandb.finish()
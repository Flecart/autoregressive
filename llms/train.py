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

# Initialize wandb
use_wandb = False
if use_wandb:
    wandb.init(project='my_project')

# Model parameters
vocab_size = 14
block_size = 512
n_embd = 128
n_head = 8
n_layer = 6

# Training parameters
num_epochs = 10
batch_size = 1024
num_workers = 20


# Instantiate the model
model = SimpleDecoderTransformer(vocab_size, block_size, n_embd, n_head, n_layer)

# If multiple GPUs are available, wrap model with nn.DataParallel
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 1], output_device=0, dim=0)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def collate_fn(batch):
    inputs, targets, mask = zip(*batch)
    # Pad sequences with 0s to the right
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    mask = pad_sequence(mask, batch_first=True, padding_value=0)
    return inputs, targets, mask


# When creating the DataLoader, pass the collate_fn
tim = time()
print("Loading dataset...")
train_dataset = MathsDataset("llms/data/train.bin")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
print(f"Time taken to load dataset: {time() - tim}")
# Training loop
for epoch in tqdm(range(num_epochs)):
    for batch in train_dataloader:
        # Get input and target sequences from batch
        input_seq, target_seq, mask = batch

        # Forward pass
        output = model(input_seq)

        # Compute loss
        loss = loss_fn(output.view(-1, vocab_size), target_seq.view(-1)) * mask.view(-1)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics to wandb
        if use_wandb:
            wandb.log({"loss": loss.item()})

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Save model every 1000 epochs
    if epoch % 1000 == 0:
        torch.save(model.state_dict(), f'llms/out/model_{epoch}.pt')

    # Keep track of best model
    if epoch == 0 or loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), 'llms/out/best_model.pt')

if use_wandb:
    wandb.finish()
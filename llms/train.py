from pydantic import BaseModel
import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.model import SimpleDecoderTransformer
from . import models
from . import dataset as ds
from time import time
import os
from .models import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR
import math
import yaml
from .configs import MainConfig, TrainingConfig
torch.set_float32_matmul_precision('high')

# ALL GLOBLS HERE
ddp = None
master_process = None
raw_model = None
tokenizer = ds.Tokenizer()

# Initialize wandb
# use_wandb = True

# # Model parameters
# tokenizer = Tokenizer()
# vocab_size = tokenizer.vocab_size
# block_size = 512
# n_embd = 128
# n_head = 8
# n_layer = 6
# config = karpathy.GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)

# # Training parameters
# num_epochs = 50
# batch_size = 1024
# # steps = 990_000 / 2048 * 600 = ~293750
# lr_decay_iters = 800_000
# num_workers = 20
# gradient_accumulation_steps = 4 # not used
# warmup_steps = 5000
# learning_rate =  0.001 # 6e-4 # max learning rate
# min_lr = 6e-5
# weight_decay = 1e-1
# beta1 = 0.9
# beta2 = 0.95

# Define a lambda function for the warmup
def lr_lambda(it: int, config: TrainingConfig):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_steps:
        return config.learning_rate * it / config.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_steps) / (config.lr_decay_iters - config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# Define the loss function and optimizer

# When creating the DataLoader, pass the collate_fn

def setup_model(config: MainConfig):
    # Instantiate the model
    # model = SimpleDecoderTransformer(vocab_size, block_size, n_embd, n_head, n_layer)
    match config.architecture.type:
        case "normal":
            model: karpathy.GPT = karpathy.GPT(config.architecture)
        case "semi-auto":
            model: sa_model.SAGPT = sa_model.SAGPT(config.architecture)
        case "frequent": # same arch
            model: karpathy.GPT = karpathy.GPT(config.architecture)
        case "complete":
            model: models.CompleteGPT = models.CompleteGPT(config.architecture)
    # weights = torch.load("llms/out/old/model_56000.pt")
    # unwanted_prefix = '_orig_mod.'
    # for k,v in list(weights.items()):
    #     if k.startswith(unwanted_prefix):
    #         weights[k[len(unwanted_prefix):]] = weights.pop(k)
    # model.load_state_dict(weights)
    if config.training.compile_model:
        model = torch.compile(model)

    # set up distributed training
    # various inits, derived attributes, I/O setup
    global ddp
    global master_process
    global raw_model
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
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        # Currently we don't use gradient
        # assert gradient_accumulation_steps % ddp_world_size == 0
        # gradient_accumulation_steps //= ddp_world_size
    else:
        print("Running in single GPU/CPU mode")
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        ddp_world_size = 1

        if torch.cuda.is_available():
            device= 'cuda:0'
            model: SimpleDecoderTransformer = model.to(device)
        else:
            device = "cpu"
    optimizer = model.configure_optimizers(
        config.training.weight_decay, 
        config.training.learning_rate, 
        (config.training.beta1, config.training.beta2), 
        device
    )

    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module
        
    return model, optimizer, device

def evaluate_downstream(model: karpathy.GPT, batch, device):
    input, targets, mask = batch
    input = input.to(device)
    targets = targets.to(device)
    mask = mask.to(device)
    input = input[:100] # only do 100 for now
    equal_count = 0
    with torch.no_grad():
        for i, sample in enumerate(input):
            # remove padding and result until the first padding
            eq_index = (sample == tokenizer.encode("=")[0]).nonzero(as_tuple=True)[0]
            sample_input = sample[:eq_index + 1]
            size_difference = input.size(1) - sample_input.size(0)
            staked_input = torch.stack([sample_input])
            generation = model.generate(staked_input, max_new_tokens=size_difference, top_k=1, stop=tokenizer.eos_token_id)
            # pad the generated sequence to match with input
            # print(generation.size(), "sizeee")
            generation_padded = torch.cat([generation, torch.zeros(1, input.size(1) - generation.size(1)).to(device)], dim=1)
            # print(i)
            # print(tokenizer.decode(sample.tolist()))
            # print(tokenizer.decode(generation_padded[0].tolist()))
            if torch.all(generation_padded == sample):
                equal_count += 1
                
    return equal_count / len(input)

def compute_loss(model, batch, device):
    input_seq, target_seq, mask = batch

    # Debug prints here
    # first_debug_print = True
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
    output, loss = model(input_seq, (target_seq, mask))
    return loss, output

class EvaluationResults(BaseModel):
    val_loss: float
    val_perplexity: float
    val_downstream_accuracy: float

def evaluate_model(model, dataloader, device) -> EvaluationResults:
    model.eval()
    total_loss = 0
    #total_perplexity = 0
    total_downstream_accuracy = 0
    total_samples = 0
    for batch in dataloader:
        loss, _ = compute_loss(model, batch, device)
        _, targets, _ = batch
        targets = targets.to(device)
        #perplexity = torch.exp(loss) # raw_model.get_perplexity(output, targets)
        downstream_accuracy = evaluate_downstream(raw_model, batch, device)
        total_loss += loss * len(batch) # Since loss is averaged
        #total_perplexity += perplexity
        total_downstream_accuracy += downstream_accuracy
        total_samples += 1
    model.train()

    mean_val_loss = total_loss / total_samples
    perplexity = torch.exp(mean_val_loss)

    return EvaluationResults(
        val_loss=mean_val_loss,
        val_perplexity=perplexity,
        val_downstream_accuracy=total_downstream_accuracy / total_samples,
    )
    
def get_dataloader(config: MainConfig, split: str, shuffle: bool = True):
    match config.architecture.type:
        case "normal":
            dataset = ds.MathsDataset(split)
        case "semi-auto":
            dataset = ds.SAMathsDataset(split, config.architecture.k_regressivity)
        case "frequent":
            dataset = ds.FreqMathsDataset(split, config.architecture.blanks, config.training.seed)
        case "complete":
            dataset = ds.MathsDataset(split)
        case _:
            raise ValueError("Invalid model type")

    return DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle, num_workers=1, pin_memory=False) # 

def train(config: MainConfig):
    torch.random.manual_seed(config.training.seed)
    model, optimizer, device = setup_model(config)
    lr_lambda_config = lambda it: lr_lambda(it, config.training)
    scheduler = LambdaLR(optimizer, lr_lambda_config)
    num_epochs = config.training.num_epochs
    
    train_dataloader = get_dataloader(config, "train")
    val_dataloader = get_dataloader(config, "val", shuffle=False)

    pbar = tqdm(total=num_epochs*len(train_dataloader))
    best_loss = 10
    steps = 0  # should save and get from saved model later.

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            _, target_seq, _ = batch
            # Get input and target sequences from batch
            loss, _ = compute_loss(model, batch, device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if steps % 500 == 0 and master_process:
                    model.eval()
                    _, target_seq, _ = batch
                    target_seq = target_seq.to(device)
                    train_downstream_accuracy = evaluate_downstream(raw_model, batch, device)
                    evaluation_results = evaluate_model(model, val_dataloader, device)
                    model.train()
                    
                    if evaluation_results.val_loss < best_loss:
                        best_loss = evaluation_results.val_loss
                        path = os.path.join(config.training.out_dir, 'best_model.pt')
                        torch.save(model.state_dict(), path)

            debug_learning_rate = lr_lambda_config(steps)
            info = {
                "loss": loss.item(), 
                "learning_rate": debug_learning_rate, 
                "train_downstream_accuracy": train_downstream_accuracy,
                **evaluation_results.model_dump()
            }

            if config.training.use_wandb and master_process:
                wandb.log(info)

            steps += 1
            if steps % 100 == 0 and master_process:
                # print(f'Step [{steps}], Loss: {loss.item()}, Val_loss: {val_loss.item()}, LR: {debug_learning_rate}, Perplexity: {perplexity}, Perplexity_train: {perplexity_train}')
                print(info)
                
            if steps % 8000 == 0 and master_process:
                path = os.path.join(config.training.out_dir, f'model_{steps}.pt')
                torch.save(model.state_dict(), path)

            pbar.set_description(f'Loss: {loss.item()}')
            pbar.update(1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


    if config.training.use_wandb and master_process:
        wandb.finish()

    if ddp:
        destroy_process_group()

def parse_and_validate_config(config_path: str) -> MainConfig:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return MainConfig(**config_data)

def handle_config() -> MainConfig:
    import argparse
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', type=str, help='Path to the config file')
    # add optional architecture params to override main config
    parser.add_argument('--vocab_size', type=int, help='Vocab size')
    parser.add_argument('--block_size', type=int, help='Block size')
    parser.add_argument('--n_embd', type=int, help='Embedding size')
    parser.add_argument('--n_head', type=int, help='Number of heads')
    parser.add_argument('--n_layer', type=int, help='Number of layers')
    parser.add_argument('--type', type=str, help='Model type')
    parser.add_argument('--k_regressivity', type=int, help='K semi regressivity parameter regressivity')
    parser.add_argument('--blanks', type=float, help='Number of blanks for frequent supervision')
    parser.add_argument('--n_masks_complete', type=int, help='Number of classes to blank for complete supervision')
    
    # training params
    parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, help='Use wandb')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, help='Compile model')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr_decay_iters', type=int, help='Learning rate decay iterations')
    parser.add_argument('--num_workers', type=int, help='Number of workers')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, help='Warmup steps')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--beta1', type=float, help='Beta1')
    parser.add_argument('--beta2', type=float, help='Beta2')
    
    dict_vars = vars(parser.parse_args())
    config_path = dict_vars.pop('config')
    config = parse_and_validate_config(config_path)
    
    architecture_keys = karpathy.GPTConfig.model_fields
    training_keys = TrainingConfig.model_fields
    
    config_dict = config.model_dump()
    for key in dict_vars:
        if dict_vars[key] is not None:
            print(f"Overriding {key} with {dict_vars[key]}")
            if key in architecture_keys:
                config_dict['architecture'][key] = dict_vars[key]
            elif key in training_keys:
                config_dict['training'][key] = dict_vars[key]
    return MainConfig(**config_dict)

def main():
    config = handle_config()
    print(f"Current configuration: {config}")
    if config.training.use_wandb:
        wandb.init(project='owt', config=config.model_dump())
        
    os.makedirs(config.training.out_dir, exist_ok=True)    
    
    train(config)
    
if __name__ == "__main__":
    main()

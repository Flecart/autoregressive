# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py # changed default, to keep the batch size the same.

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 18 batch size * 1024 block size * 5 gradaccum * 3 = 491,520
# But in our case we don't have enough memory for single GPU for this.
# Model takes 124M * 4bytes to store in 32bit precision, so 0.5GB per GPU.
# Backward pass takes 2x memory, so 1GB per GPU.
# Empirically we see that batch_size 18 is the maximum we can have
batch_size = 18
block_size = 1024
gradient_accumulation_steps = 5 * 3

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py # changed default, to keep the batch size the same.

wandb_log = True
wandb_project = 'owt'
wandb_run_name='small-gpt-maths'

# these make the total batch size be ~0.5M
# 256 batch size * 512 block size * 4 = 524 288 tokens per batch
batch_size = 256
# block_size = 1024
gradient_accumulation_steps = 4

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

dataset = 'maths'
vocab_size = 14
block_size = 512 
n_embd = 128
n_head = 8
n_layer=  6 # circa 1kk parametri.
meta_vocab_size = 14
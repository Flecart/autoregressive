# This file is just used to quickly launch training instances, just toggle the one you like :D
# Remember to ACTIVATE your poetry env

# cd gpt-2

# run standard gpt2 run, as per https://github.com/karpathy/nanoGPT?tab=readme-ov-file#reproducing-gpt-2
# torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py "$@"

# run gpt2 sa with k=2
# if debugging, run with --wandb_log=False --compile=False,
# torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_sa.py "$@"
# torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_maths.py "$@"
# torchrun --standalone --nproc_per_node=4 -m llms.train

# CUDA_VISIBLE_DEVICES=1 python3 -m llms.train llms/configs/main.yaml
# CUDA_VISIBLE_DEVICES=1 python3 -m llms.train llms/configs/main.yaml
CUDA_VISIBLE_DEVICES=1 python3 -m llms.train llms/configs/main.yaml --k_regressivity=2 --use_wandb --compile --out_dir="llms/out/sa-2"
# This file is just used to quickly launch training instances, just toggle the one you like :D

cd gpt-2

# run standard gpt2 run, as per https://github.com/karpathy/nanoGPT?tab=readme-ov-file#reproducing-gpt-2
# poetry run torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py "$@"

# run gpt2 sa with k=2
poetry run torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_sa.py "$@"
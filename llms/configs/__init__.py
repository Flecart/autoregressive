from typing import Optional, Literal
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    num_epochs: int
    batch_size: int
    lr_decay_iters: int
    num_workers: int
    gradient_accumulation_steps: Optional[int] = None  # Mark optional fields with default values
    warmup_steps: int
    learning_rate: float
    min_lr: float
    weight_decay: float
    beta1: float
    beta2: float
    out_dir: str
    compile_model: bool = False
    use_wandb: bool = False
    
class GPTConfig(BaseModel):
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    type: Literal["normal", "semi-auto"] = "normal"
    
    # Used for semi-autoregressive models
    k_regressivity: int = 2

class MainConfig(BaseModel):
    architecture: GPTConfig
    training: TrainingConfig
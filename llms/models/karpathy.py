"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from abc import abstractmethod
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from .model import Abacus
from ..tokenizer import Tokenizer
from ..configs import GPTConfig
from pydantic import BaseModel

tokenizer = Tokenizer()

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head: int, dropout: float, bias=True):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            raise NotImplementedError("Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
            #                             .view(1, 1, block_size, block_size))

    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            raise NotImplementedError("Flash Attention requires PyTorch >= 2.0")
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, n_embd: int, bias=True, dropout=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias=True):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias=bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MetaGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
    
    def init_weights(self):    
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    @torch.no_grad
    def get_perplexity(self, logits, targets):
        """ Returns perplexity

        Input:
            logits: Shape: (batch_size, sequence_length, vocab_size)
            Targets: Shape: (batch_size, sequence_length)
        
        """
        # far controllare a Samu
        logit_predictions = F.log_softmax(logits, dim=-1)
        one_hot_targets = F.one_hot(targets, num_classes=self.config.vocab_size).float()
        perplexity = torch.exp(-torch.sum(logit_predictions * one_hot_targets, dim=-1).mean())
        return perplexity

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
            # n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of 242 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 242e12 # L4 GPU bfloat16 peak flops is 242 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def print_probs(self, output):
        # Prints probability vectors nicely
        for j in range(output.size(1)):
            print(f"{tokenizer._convert_id_to_token(j)}: {output[0, j]:.3f}", end=" ")
        print()

        output = torch.argmax(output, dim=-1)
        return output
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

class GPT(MetaGPT):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = Abacus(config.n_embd),  # nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config.n_embd, config.n_head, config.dropout, bias=config.bias) 
                for _ in range(config.n_layer)]
            ),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        print("Number of parameters of a single transformer block:", sum(p.numel() for p in self.transformer.h[0].parameters()))
        print("Number of wte parameters:", sum(p.numel() for p in self.transformer.wte.parameters()))
        print("Number of wpe parameters:", sum(p.numel() for p in self.transformer.wpe.parameters()), config.vocab_size, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, attention_mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        print(f"Size idx: {idx.size()}")
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(idx)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            targets, masks = targets
            
            # logits = torch.stack([head_output[:, i::self.config.k_regressivity, :] for i in range(self.config.k_regressivity)]) # (k, b, t, vocab_size)
            # stacked_targets = torch.stack([targets[:, i:i+self.config.block_size][:, i::self.config.k_regressivity] for i in range(self.config.k_regressivity)], ) # (k, b, t)
            logits = self.lm_head(x) # (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none") * masks.view(-1)
            loss = loss.mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = torch.stack([lm_head(x[:, [-1], :]) for lm_head in self.lm_heads]) # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, -self.config.k_regressivity:, :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int=20, temperature: float=1.0, top_k=None, stop=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.size(0) == 1, "can only work with a single example at a time for now"
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if stop is not None and idx_next.item() == stop:
                break

        return idx

    # 
    # @torch.no_grad()
    # def ____generate(self, idx, max_new_tokens: int=20, temperature: float=1.0, top_k=None, stop=None):
    #     """
    #     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    #     the sequence max_new_tokens times, feeding the predictions back into the model each time.
    #     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    #     """
    #     stop_flags = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    #     initial_lenght = idx.size(1)
    #     # find the start of the padding token or the end of the sequence for correct generation
    #     start_or_eos = torch.ones(idx.size(0), device=idx.device) * idx.size(1)
        
    #     # loop way
    #     # for i in range(idx.size(0)):
    #     #     # find the first padding token if any
    #     #     indexes = (idx[i] == tokenizer.vocab[tokenizer.pad_token]).nonzero()
    #     #     if indexes.size(0) > 0:
    #     #         start_or_eos[i, indexes[0, 0]] = 0
        
    #     indexes = (idx == tokenizer.vocab[tokenizer.pad_token]).nonzero()
    #     if indexes.size(0) > 0:
    #         start_or_eos[indexes[:, 1]] = 0
            
    #     for _ in range(max_new_tokens):
    #         # if the sequence context is growing too long we must crop it at block_size
    #         idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    #         # forward the model to get the logits for the index in the sequence
    #         logits = self(idx_cond)
    #         # pluck the logits at the final step and scale by desired temperature
    #         logits = logits[:, start_or_eos, :] / temperature
    #         # optionally crop the logits to only the top k options
    #         self.print_probs(F.softmax(logits, dim=-1)) 
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float('Inf')
    #         probs = F.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         stop_flags = stop_flags | (idx_next.squeeze(-1) == stop)
            
    #         mask_lower_eos = start_or_eos.unsqueeze(-1) < idx.size(1)
    #         idx_next = idx_next.masked_fill(stop_flags.unsqueeze(-1) | mask_lower_eos, tokenizer.vocab[tokenizer.pad_token])
            
    #         # The loop classical way            
    #         for i in range(idx.size(0)):
    #             if start_or_eos[i] != idx.size(1):
    #                 idx[i, start_or_eos[i]] = idx_next[i]
    #                 start_or_eos[i] += 1
                    
    #         if start_or_eos.all() == idx.size(1):
    #         idx = torch.cat((idx, idx_next), dim=1)
                    
    #         mask = start_or_eos != idx.size(1)
    #         idx = torch.where(mask.unsqueeze(-1), idx.scatter(1, start_or_eos.unsqueeze(-1), idx_next), idx)
    #         start_or_eos = torch.where(mask, start_or_eos + 1, start_or_eos)
    #         idx = torch.cat((idx, idx_next), dim=1)
                    
    #         if stop is not None and stop_flags.sum() == idx.size(0):
    #             break

    #     return idx

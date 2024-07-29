from .karpathy import GPT, GPTConfig

import torch
from torch.nn import functional as F

class CompleteGPT(GPT):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        
        self.n_masks_complete = config.n_masks_complete
    
    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
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
            logits: torch.Tensor = self.lm_head(x) # (b, t, vocab_size)
            
            one_hot_masks = torch.zeros(b, t, self.config.vocab_size, device=logits.device, dtype=torch.bool)
            
            # for each b,t sample without reimmission n_masks_complete times, sample a mask
            uniform_ones = torch.ones(b, t, self.config.vocab_size, device=logits.device)
            target_one_hot = torch.nn.functional.one_hot(targets, logits.size(-1))
            uniform_ones -= target_one_hot # so that we don't deactivate the target.
            masks_complete = torch.multinomial(uniform_ones.view(-1, uniform_ones.size(-1)), self.n_masks_complete).view(b, t, self.n_masks_complete)
            
            for i in range(self.n_masks_complete):
                one_hot_masks |= torch.nn.functional.one_hot(masks_complete[:, :, i], logits.size(-1)).bool()
            # set every sampled vocab size to -infty
            logits = logits.masked_fill(one_hot_masks == 1, -float('inf'))
            
            loss: torch.Tensor = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
            loss = (loss.view(b, t) * masks).sum(dim=-1) / masks.sum(dim=-1) # media singolo case, con peso corretto data dalla mask.
            loss = loss.mean()  # media lungo il batch
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = torch.stack([lm_head(x[:, [-1], :]) for lm_head in self.lm_heads]) # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, -self.config.k_regressivity:, :])
            loss = None

        return logits, loss
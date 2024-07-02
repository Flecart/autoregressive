import torch.nn as nn

from . import karpathy
from .. import configs
import torch
from torch.nn import functional as F

class SACausalSelfAttention(karpathy.CausalSelfAttention):
    def __init__(self, config: configs.GPTConfig):
        super().__init__(n_embd=config.n_embd, n_head=config.n_head, dropout=config.dropout, bias=config.bias)
        self.k_regressivity = config.k_regressivity

    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            L = (q.size(-2) + self.k_regressivity - 1) // self.k_regressivity
            S = (k.size(-2) + self.k_regressivity - 1) // self.k_regressivity
            # print(f"q.size()={q.size()} k.size()={k.size()} v.size()={v.size()} L={L} S={S}")
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)
            temp_mask = temp_mask.repeat_interleave(self.k_regressivity, dim=1).repeat_interleave(self.k_regressivity, dim=0) # duplicate all true keys for k_regressivity!
            temp_mask = temp_mask[:q.size(-2), :k.size(-2)] # crop to size
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=temp_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            raise NotImplementedError("Causal attention not implemented for non-flash attention")
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class SAGPTBlock(karpathy.Block):
    def __init__(self, config: configs.GPTConfig):
        super().__init__(n_embd=config.n_embd, n_head=config.n_head, dropout=config.dropout, bias=config.bias)
        self.attn = SACausalSelfAttention(config)

class SAGPT(karpathy.MetaGPT):
    def __init__(self, config: configs.GPTConfig):
        super().__init__(config)
        print("Using semi-autoregressive GPT model")
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = karpathy.Abacus(config.n_embd),  # nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([SAGPTBlock(config) for _ in range(config.n_layer)]
            ),
            ln_f = karpathy.LayerNorm(config.n_embd, bias=config.bias),
        ))

        print("Number of parameters of a single transformer block:", sum(p.numel() for p in self.transformer.h[0].parameters()))
        print("Number of wte parameters:", sum(p.numel() for p in self.transformer.wte.parameters()))
        print("Number of wpe parameters:", sum(p.numel() for p in self.transformer.wpe.parameters()), config.vocab_size, config.n_embd)
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(config.k_regressivity)])

        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_heads[0].weight # https://paperswithcode.com/method/weight-tying
        self.init_weights()

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(idx) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # (b, t, n_embd)
        
        if targets is not None:
            targets, masks = targets
            final_size = (targets.size(1) + self.config.k_regressivity - 1) // self.config.k_regressivity
            
            final_outputs = []
            targets_to_stack = []
            masks_to_stack = []
            for i, lm_head in enumerate(self.lm_heads):
                logits: torch.Tensor = lm_head(x)[:, i::self.config.k_regressivity]
                curr_target: torch.Tensor = targets[:, i::self.config.k_regressivity]
                curr_masks: torch.Tensor = masks[:, i::self.config.k_regressivity]
                if logits.size(1) < final_size:
                    padding = torch.zeros((b, final_size - logits.size(1), logits.size(2)), dtype=logits.dtype ,device=device)
                    padding_target = torch.zeros((b, final_size - logits.size(1)), dtype=curr_target.dtype, device=device)
                    logits = torch.cat((logits, padding), dim=1)
                    curr_target = torch.cat((curr_target, padding_target), dim=1)
                    curr_masks = torch.cat((curr_masks, padding_target), dim=1)
                    
                # print(f"lm_head={i} logits.size()={logits.size()} curr_target.size()={curr_target.size()}")
                final_outputs.append(logits)
                targets_to_stack.append(curr_target)
                masks_to_stack.append(curr_masks)
                # (k, b, t, vocab_size)
            logits = torch.stack(final_outputs) # torch.stack([self.lm_heads[i](x)[:, i::self.config.k_regressivity] for i in range(self.config.k_regressivity)]) 
            stacked_targets = torch.stack(targets_to_stack) # torch.stack([targets[:, i::self.config.k_regressivity] for i in range(self.config.k_regressivity)], ) # (k, b, t)
            stacked_masks = torch.stack(masks_to_stack) # torch.stack([masks[:, i::self.config.k_regressivity] for i in range(self.config.k_regressivity)], ) # (k, b, t)
            # logits = self.lm_head(x) # (b, t, vocab_size)
            loss: torch.Tensor = F.cross_entropy(logits.view(-1, logits.size(-1)), stacked_targets.view(-1), reduction="none") * stacked_masks.view(-1)
            loss = loss.mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = torch.stack([lm_head(x[:, [-1], :]) for lm_head in self.lm_heads]) # note: using list [-1] to preserve the time dim
            # logits = self.lm_head(x[:, -self.config.k_regressivity:, :])
            loss = None

        return logits, loss

    @torch.no_grad
    def get_perplexity(self, logits, targets):
        """ Returns perplexity

        Input:
            logits: Shape: (batch_size, sequence_length, vocab_size)
            Targets: Shape: (batch_size, sequence_length)
        
        """
        final_size = (targets.size(1) + self.config.k_regressivity - 1) // self.config.k_regressivity
        
        targets_to_stack = []
        for i in range(self.config.k_regressivity):
            curr_target: torch.Tensor = targets[:, i::self.config.k_regressivity]
            if curr_target.size(1) < final_size:
                padding_target = torch.zeros((targets.size(0), final_size - curr_target.size(1)), dtype=curr_target.dtype, device=targets.device)
                curr_target = torch.cat((curr_target, padding_target), dim=1)
            targets_to_stack.append(curr_target)
            
        stacked_targets = torch.stack(targets_to_stack) # (k, b, t)
        
        print(f"aaaaaaaaaaaa", stacked_targets)
        loss        = F.cross_entropy(logits.view(-1, logits.size(-1)), stacked_targets.view(-1), reduction="mean")
        print(f"bbbbbbbbbbbb", loss)
        perplexity  = torch.exp(loss)
        
        # logit_predictions = F.log_softmax(logits, dim=-1)
        # one_hot_targets = F.one_hot(stacked_targets, num_classes=self.config.vocab_size).float()
        # perplexity = torch.exp(-torch.sum(logit_predictions * one_hot_targets, dim=-1).mean())
        return perplexity

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature=1.0, top_k=None, stop=None):
        """
        """
        stopped = False
        tokens = 0
        while True:
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Below here is present the old approach that used multiple heads to predict multiple tokens at the same time.
            for k in range(self.config.k_regressivity):
                # pluck the logits at the final step and scale by desired temperature
                logits_curr = logits[k, :, -1, :] / temperature # (k-reg, batch, lunghezza-stringa, vocab_size)

                if top_k is not None:
                    v, _ = torch.topk(logits_curr, min(top_k, logits_curr.size(-1)))
                    logits_curr[logits_curr < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits_curr to (normalized) probabilities
                probs = F.softmax(logits_curr, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)
                tokens += 1
                if (tokens >= max_new_tokens) or (stop is not None and idx_next.item() == stop):
                    stopped = True
                    break
            if stopped:
                break
            # print(f"current size is {idx.size()}")
        return idx
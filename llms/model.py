import torch
import torch.nn as nn
import torch.optim as optim
from .tokenizer import Tokenizer
import random
tokenizer = Tokenizer()

"""Implementation of abacus embeddings"""
# Example of how to extract digit tokens to pass into constructor
class Abacus(torch.nn.Module):
    """
    Abacus Embeddings, learned emebddings resued for each digit.
    Integers must be reversed for this to work correctly.
    Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
    """
    def __init__(self, embedding_dim, max_seq_length=1024, max_k=99):
        """
        digit_tokens (list): list of the tokens for each of the 10 digits, `digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])`
        embedding_dim (int): dimension to embed into
        max_seq_length (int): maximum number of embeddings that can be trained
        max_k (int): maximum k value which we randomly shift by during training
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)

        self.max_k = max_k

    def helper(self, mask, device):
        """
        Converts a binary mask of digit locations into spans of consecutive digits
        """
        mask_shape = mask.shape
        
        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat([torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype), mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask
        
        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)
        
        # Generate an index array row-wise
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)
        
        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)
        
        # Calculate positions in segment
        positions = index - reset_index.gather(1, segment_ids) + 1
        
        # Ensure only values within 1-segments are non-zero
        result = positions * mask

        return result

    def forward(self, input_ids):
        """
        input_ids (tensor): a batch of inputs, each row is a sample
        """
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)

        k=0
        if self.training:
            k = random.randint(0, self.max_k)
            output[output>0] += k # as we already have ones in the tensor, the tensor values will be k+1

        return self.embedding(output)


class SimpleDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super(SimpleDecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        # self.pos_embed = nn.Embedding(block_size, n_embd)
        self.pos_embed = Abacus(n_embd)
        self.layers: list[nn.TransformerDecoderLayer] = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd,  norm_first=True) for _ in range(n_layer)
        ])

        self.fc_out = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_length = x.size(1)
        # positions = torch.arange(0, seq_length).unsqueeze(0).to(x.device)
        
        # x = self.embed(x) + self.pos_embed(positions)
        x = self.embed(x) + self.pos_embed(x)
        
        # memory_mask = torch.ones(x.size(0), x.size(0), dtype=torch.bool).to(x.device)
        causal_attention_mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1).to(x.device)
        for layer in self.layers:
            # ignore the memory layer, add a mask
            x = layer(x, x, tgt_is_causal=True, tgt_mask=causal_attention_mask) #, memory_mask=memory_mask, )
        
            # print("Inside forward", x[0])

        return self.fc_out(x)

    def print_probs(self, output):
        # Prints probability vectors nicely
        for j in range(output.size(2)):
            print(f"{j}: {output[0, -1, j]:.2f}", end=" ")
        print()

        output = torch.argmax(output, dim=-1)
        return output

    def generate(self, x, max_len=10, stop_token=13):
        for _ in range(max_len):
            output = self.forward(x)
            output = torch.softmax(output, dim=-1)
            self.print_probs(output)
            output = torch.argmax(output, dim=-1)
            x = torch.cat([x, output[:, -1].unsqueeze(-1)], dim=-1)
            
            if output[0, -1] == stop_token:
                break
        
        return x

# Model parameters
# vocab_size = 14
# block_size = 512
# n_embd = 128
# n_head = 8
# n_layer = 6

# # Instantiate the model
# model = SimpleTransformer(vocab_size, block_size, n_embd, n_head, n_layer)

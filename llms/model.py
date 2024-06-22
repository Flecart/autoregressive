import torch
import torch.nn as nn
import torch.optim as optim


class SimpleDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super(SimpleDecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.layers: list[nn.TransformerDecoderLayer] = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd,  norm_first=True) for _ in range(n_layer)
        ])
        self.fc_out = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length).unsqueeze(0).to(x.device)
        
        x = self.embed(x) + self.pos_embed(positions)
        
        memory_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool).to(x.device)
        for layer in self.layers:
            # ignore the memory layer, add a mask
            x = layer(x, x, tgt_is_causal=True, memory_mask=memory_mask)
        
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

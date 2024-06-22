import torch
import torch.nn as nn
import torch.optim as optim


class SimpleDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super(SimpleDecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=n_embd, nhead=n_head) for _ in range(n_layer)
        ])
        self.fc_out = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length).unsqueeze(0).to(x.device)
        
        x = self.embed(x) + self.pos_embed(positions)
        
        for layer in self.layers:
            x = layer(x, x)
        
        return self.fc_out(x)

# Model parameters
# vocab_size = 14
# block_size = 512
# n_embd = 128
# n_head = 8
# n_layer = 6

# # Instantiate the model
# model = SimpleTransformer(vocab_size, block_size, n_embd, n_head, n_layer)

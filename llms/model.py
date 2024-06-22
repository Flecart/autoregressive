import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimpleDecoderTransformer(pl.LightningModule):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, learning_rate=1e-3):
        super(SimpleDecoderTransformer, self).__init__()
        self.save_hyperparameters()
        
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=n_embd, nhead=n_head) for _ in range(n_layer)
        ])
        self.fc_out = nn.Linear(n_embd, vocab_size)
        self.learning_rate = learning_rate

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length).unsqueeze(0).to(x.device)
        
        x = self.embed(x) + self.pos_embed(positions)
        
        for layer in self.layers:
            x = layer(x, x)
        
        return self.fc_out(x)

    def generate(self, x, max_len=100, stop_token=13):
        for _ in range(max_len):
            output = self.forward(x)
            output = torch.argmax(output, dim=-1)
            x = torch.cat([x, output[:, -1].unsqueeze(-1)], dim=-1)
            
            if output[0, -1] == stop_token:
                break
        
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.hparams.vocab_size), y.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Example usage:
# from pytorch_lightning import Trainer

# model = SimpleDecoderTransformerPL(vocab_size=14, block_size=512, n_embd=128, n_head=8, n_layer=6)
# trainer = Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader)
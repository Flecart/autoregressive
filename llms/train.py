import pytorch_lightning as pl
from argparse import ArgumentParser
from .model import SimpleDecoderTransformer
import yaml
from pydantic import BaseModel

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class TransformerParams(BaseModel):
    vocab_size: int
    block_size: int
    emb_size: int
    n_head: int
    n_layer: int

class TrainParams(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the YAML configuration file", default="")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--vocab_size", type=int, default=14)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=6)

    args = parser.parse_args()

    config = load_config(args.config)

    transformer_params = TransformerParams(**config["architecture"])
    train_params = TrainParams(**config["training"])

    model = SimpleDecoderTransformer(
        vocab_size=transformer_params.vocab_size,
        block_size=transformer_params.block_size,
        n_embd=transformer_params.emb_size,
        n_head=transformer_params.n_head,
        n_layer=transformer_params.n_layer,
        learning_rate=train_params.learning_rate
    )

    trainer = pl.Trainer()
    trainer.fit(model)
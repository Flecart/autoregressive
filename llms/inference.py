import torch
from .model import SimpleDecoderTransformer
from .dataset import Tokenizer
from . import karpathy

# Assuming the tokenizer and the model's state_dict are already loaded
tokenizer = Tokenizer()

# Model parameters
vocab_size = tokenizer.vocab_size
block_size = 512
n_embd = 128
n_head = 8
n_layer = 6
config = karpathy.GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)

# Instantiate the model
# model = SimpleDecoderTransformer(vocab_size, block_size, n_embd, n_head, n_layer)
model = karpathy.GPT(config)

# inspect the state dict
weights = torch.load("llms/out/best_model.pt")
unwanted_prefix = '_orig_mod.'
for k,v in list(weights.items()):
    if k.startswith(unwanted_prefix):
        weights[k[len(unwanted_prefix):]] = weights.pop(k)
# for k, v in weights.items():
#     print(k, v.size())

# Load the model weights
model.load_state_dict(weights)

# Set the model to evaluation mode
model.eval()

# Sample input text
input_text = "7414487100717341052+2273996284028="

# Convert input text to token IDs using the tokenizer
# This step depends on the tokenizer you're using
token_ids = tokenizer.encode(input_text)
# add bos to beginning
token_ids = [tokenizer.bos_token_id] + token_ids

# Convert token IDs to PyTorch tensor and add batch dimension
input_tensor = torch.tensor([token_ids], dtype=torch.long)

# Perform inference
with torch.no_grad():
    generated = model.generate(input_tensor)


print(generated, generated.size())
# Convert predicted token IDs back to text
predicted_text = tokenizer.decode(generated[0].tolist())

print(predicted_text)
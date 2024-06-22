import torch
from .model import SimpleDecoderTransformer
from .dataset import Tokenizer

# Assuming the tokenizer and the model's state_dict are already loaded

# Model parameters
vocab_size = 14
block_size = 512
n_embd = 128
n_head = 8
n_layer = 6

# Instantiate the model
model = SimpleDecoderTransformer(vocab_size, block_size, n_embd, n_head, n_layer)

# Load the model weights
model.load_state_dict(torch.load("llms/out/best_model.pt"))

# Set the model to evaluation mode
model.eval()

# Sample input text
input_text = "1234 + 14 = "

# Convert input text to token IDs using the tokenizer
# This step depends on the tokenizer you're using
tokenizer = Tokenizer()
token_ids = tokenizer.encode(input_text)

# Convert token IDs to PyTorch tensor and add batch dimension
input_tensor = torch.tensor([token_ids], dtype=torch.long)

# Perform inference
with torch.no_grad():
    generated = model.generate(input_tensor)


print(generated, generated.size())
# Convert predicted token IDs back to text
predicted_text = tokenizer.decode(generated[0].tolist())

print(predicted_text)
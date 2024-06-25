import torch
from .models.model import SimpleDecoderTransformer
from .dataset import Tokenizer
from .models import karpathy

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
weights = torch.load("llms/out/old/model_56000.pt")
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

# Train
# 

# Test
# 51190348+47347=98438348XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# 15+84945565=99945565XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# 206529+050878611=256308711XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# 206529+050878611=256308711XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# 908285621305640176+169701030186686092=077096651481337169XXXXXX
# 9314+763623=605033XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Sample input text
input_text = "9314+763623=" # 077096651481337169XXXXXX

# Convert input text to token IDs using the tokenizer
# This step depends on the tokenizer you're using
token_ids = tokenizer.encode(input_text)
# add bos to beginning
token_ids = [tokenizer.bos_token_id] + token_ids
second_text = [tokenizer.bos_token_id] +[tokenizer.encode("206529+050878611=")]

# Convert token IDs to PyTorch tensor and add batch dimension
input_tensor = torch.tensor([token_ids, second_text], dtype=torch.long)

# Perform inference
with torch.no_grad():
    generated = model.generate(input_tensor, max_new_tokens=50, top_k=1, stop=tokenizer.eos_token_id)


print(generated, generated.size())
# Convert predicted token IDs back to text
predicted_text = tokenizer.decode(generated[0].tolist())
predicted_text2 = tokenizer.decode(generated[1].tolist())

print(predicted_text)
print(predicted_text2)
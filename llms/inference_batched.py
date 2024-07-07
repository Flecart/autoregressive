import torch
from .models.model import SimpleDecoderTransformer
from .dataset import Tokenizer
from .models import karpathy, sa_model

# Assuming the tokenizer and the model's state_dict are already loaded
tokenizer = Tokenizer()

# Model parameters
vocab_size = tokenizer.vocab_size
block_size = 512
n_embd = 128
n_head = 8
n_layer = 6
config = karpathy.GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, k_regressivity=1)

# Instantiate the model
# model = SimpleDecoderTransformer(vocab_size, block_size, n_embd, n_head, n_layer)
model = karpathy.GPT(config)
# model: sa_model.SAGPT = sa_model.SAGPT(config)

# inspect the state dict
# weights = torch.load("llms/out/old/model_56000.pt")
# weights = torch.load("llms/out/old-sa/sa-1/model_472000.pt")
# unwanted_prefix = '_orig_mod.'
# for k,v in list(weights.items()):
#     if k.startswith(unwanted_prefix):
#         weights[k[len(unwanted_prefix):]] = weights.pop(k)
# for k, v in weights.items():
#     print(k, v.size())

# Load the model weights
# model.load_state_dict(weights)

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
input_text = "9314+763623=605033" # 077096651481337169XXXXXX
target_text = "9314+763623=605033"


# Convert input text to token IDs using the tokenizer
# This step depends on the tokenizer you're using
token_ids = tokenizer.encode(input_text)
target_ids = tokenizer.encode(target_text)
# add bos to beginning
token_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
target_ids = target_ids + [tokenizer.eos_token_id, tokenizer.pad_token_id]
# second_text = [tokenizer.bos_token_id] +[tokenizer.encode("206529+050878611=")]

token_ids = torch.tensor([token_ids], dtype=torch.int64)
target_ids = torch.tensor([target_ids], dtype=torch.int64)

mask = torch.ones((1, len(token_ids)), dtype=torch.int64)

eos_token = tokenizer.vocab[tokenizer.eos_token]
equal_token = tokenizer.vocab["="]
index_of_eot = (token_ids == eos_token).nonzero(as_tuple=True)[0]
index_of_eq = (token_ids == equal_token).nonzero(as_tuple=True)[0]
mask[0, :index_of_eq + 1] = 0
mask[0, index_of_eot + 1:] = 0

numpy_input = token_ids.numpy()
print(tokenizer.decode(numpy_input[0]))
mask_input = mask.numpy()
for x in mask_input[0]:
    print(x, end="")
print()
numpy_target = target_ids.numpy()
print(tokenizer.decode(numpy_target[0]))
print(token_ids.size())

# Convert token IDs to PyTorch tensor and add batch dimension
# input_tensor = torch.tensor([token_ids], dtype=torch.long)
input_tensor = torch.cat([token_ids, token_ids], dim=1)

# Perform inference
with torch.no_grad():
    generated = model.generate(input_tensor, max_new_tokens=50, top_k=1, stop=tokenizer.eos_token_id)

    # logits, loss = model(input_tensor, (target_ids, mask))
    
    # perplexity = torch.exp(loss)
    
# print(f"Perplexity: {perplexity}")
# # print(f"Generated: {generated}"
# print(f"Loss: {loss}")

# current_logit = logits[0, 0, -2, :]
# print(f"Logits: {current_logit}")

# # print(f"shape logits: {logits.size()}")

# probabilities = torch.softmax(current_logit, dim=-1)
# print(f"Probabilities: {probabilities.size()}")
# print(f"Current Logit: {current_logit.size()}")
# # print dictionary of values of the given probabilities
# for i in range(current_logit.size(0)):
#     char = tokenizer.decode([i])
#     print(f"{char}: {probabilities[i].item():.2f}", end=" ")
# print()

# print(generated, generated.size())
# Convert predicted token IDs back to text
# predicted_text = tokenizer.decode(generated[0].tolist())
# predicted_text2 = tokenizer.decode(generated[1].tolist())

# print(predicted_text)
# print(predicted_text2)
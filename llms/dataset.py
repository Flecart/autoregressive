# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from pydantic import BaseModel
from datasets import DatasetDict, Dataset
import torch

class Operation(BaseModel):
    operand1: int
    operand2: int
    operator: str = "+"

    def __str__(self):
        return f"{self.operand1} {self.operator} {self.operand2} = {self.call()[::-1]}" # we reverse it bc probably easier.
    
    def call(self):
        return str(eval(f"{self.operand1} {self.operator} {self.operand2}"))

# from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 20

dataset_padding = 30
# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# enc = tiktoken.get_encoding("gpt2")

class MathsDataset(Dataset):
    def __init__(self, filepath: str):
        self.__tokenizer = Tokenizer()
        # too slow to load the whole dataset
        self.data_source = np.memmap(filepath, dtype=np.uint16, mode='r')
        
        # values = self.__tokenizer.decode(data)
        # samples = values.split("\n")
        # # Now we can create the dataset

        # self.data = [self.__tokenizer.encode(sample) for sample in samples]
        # self.masks = []
        # for sample in samples:
        #     # index of the string '=' in the sample
        #     index = sample.index("=")
        #     mask = np.zeros(len(sample))
        #     mask[index + 2:] = 1
        #     self.masks.append(mask)

    def __len__(self):
        return 990_000 # from the generation, should be change to
        # return len(self.data)

    def __getitems__(self, idx: list[int]):
        """ idx: list of indices to retrieve from the dataset
        """
        # print(type(idx), idx)
        pad_starts = np.array(idx) * dataset_padding
        samples = torch.stack([torch.from_numpy((self.data_source[pad_start : pad_start + dataset_padding]).astype(np.int64)) for pad_start in pad_starts])
        mask = torch.ones((len(idx), dataset_padding), dtype=torch.int64)

        for i, sample in enumerate(samples):
            # https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
            index_of_eot = (sample == self.__tokenizer.eot_token).nonzero(as_tuple=True)[0]
            index_of_eq = (sample == 11).nonzero(as_tuple=True)[0]
            mask[i, :index_of_eq + 2] = 0
            mask[i, index_of_eot + 1:] = 0

            # Examle of the output:
            # Example: 683886 + 420998 = 4884011X0000
            # Example: 000000000000000000111111110000
            # Example: 406146 + 66478 = 426274X000000
            # Example: 000000000000000001111111000000
            # print("Example:", self.__tokenizer.decode(sample.numpy()))
            # numpy_mask = mask[i].numpy()
            # print("Example: ", end="")
            # for j, mask_val in enumerate(numpy_mask):
            #     print(int(mask_val), end="")
            # print()

        input_seq = samples[:, :-1]
        target_seq = samples[:, 1:]
        mask = mask[:, 1:]
        return input_seq, target_seq, mask

    def get_string(self, idx):
        return self.__tokenizer.decode(self.data[idx])

class Tokenizer():
    def __init__(self):
        self.eot_token = 13
        self.max_token_value = 13

    def encode(self, text):
        value = []
        for char in text:
            if char == " ":
                value.append(10)
            elif char == "=":
                value.append(11)
            elif char == "+":
                value.append(12)
            elif char.isnumeric():
                value.append(int(char))
            else:
                raise ValueError(f"Invalid character in text: {char}")
        return value
            
    def decode(self, l):
        string = ""
        for i in l:
            if i == 10:
                string += " "
            elif i == 11:
                string += "="
            elif i == 12:
                string += "+"
            elif i == 13:
                string += "X"
            elif i < 10:
                string += str(i)
            else:
                raise ValueError(f"Invalid token in list: {i}")
        return string

enc = Tokenizer()
if __name__ == '__main__':

    number_of_operations = 1000000
    max_number_size = 1000000
    test_split_size = 0.01
    np.random.seed(42)

    operations = []
    def gen():
        for _ in range(number_of_operations):
            op = Operation(
                operand1=np.random.randint(max_number_size),
                operand2=np.random.randint(max_number_size),
                operator=np.random.choice(["+"]), # , "-", "*", "/"
            )

            yield {"text": str(op)}
    dataset = Dataset.from_generator(gen, num_proc=num_proc_load_dataset)
    split_dataset = dataset.train_test_split(test_size=test_split_size)

    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    # owt by default only contains the 'train' split, so create a test split
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode(example['text']) # encode ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        # pad ids to 30, because it's easier to retrive in this way
        ids = ids + [0] * (dataset_padding - len(ids))
        assert(len(ids) == dataset_padding), f"len(ids) = {len(ids)}"
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), "data", f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # this was for the original data
    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')

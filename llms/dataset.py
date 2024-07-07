# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from pydantic import BaseModel
from datasets import DatasetDict, Dataset
import torch
from argparse import ArgumentParser
from .tokenizer import Tokenizer
class Operation(BaseModel):
    operand1: int
    operand2: int
    operator: str = "+"

    def __str__(self):
        return f"{self.operand1[::-1]}{self.operator}{self.operand2[::-1]}={self.call()[::-1]}" # we reverse it bc probably easier.
    
    def call(self):
        return str(eval(f"{self.operand1} {self.operator} {self.operand2}"))

# from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 80

dataset_padding = 65
# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# enc = tiktoken.get_encoding("gpt2")

# generation parameters
number_of_operations = 20_000_000
max_number_size = int(1e20) # 20 characters
test_split_size = 0.01
max_val_set = 2048

class MathsDataset(Dataset):
    def __init__(self, split: str = None):
        if split == "train":
            filepath = os.path.join(os.path.dirname(__file__), "data", "train.bin")
        elif split == "val":
            filepath = os.path.join(os.path.dirname(__file__), "data", "val.bin")
        else:
            raise ValueError(f"Invalid split: {split}")
        self._split = split
        self.tokenizer = Tokenizer()
        # too slow to load the whole dataset
        self.data_source = np.memmap(filepath, dtype=np.uint16, mode='r')
        
        # values = self.tokenizer.decode(data)
        # samples = values.split("\n")
        # # Now we can create the dataset

        # self.data = [self.tokenizer.encode(sample) for sample in samples]
        # self.masks = []
        # for sample in samples:
        #     # index of the string '=' in the sample
        #     index = sample.index("=")
        #     mask = np.zeros(len(sample))
        #     mask[index + 2:] = 1
        #     self.masks.append(mask)

    def __len__(self):
        if self._split == "train":
            return (number_of_operations // 100) * int(100 - test_split_size * 100)
        elif self._split == "val":
            return min((number_of_operations // 100) * int(test_split_size * 100), max_val_set) # faster check in this way

    def __getitems__(self, idx: list[int]):
        """ idx: list of indices to retrieve from the dataset
        """
        # print(type(idx), idx)
        pad_starts = np.array(idx) * dataset_padding
        samples = torch.stack([torch.from_numpy((self.data_source[pad_start : pad_start + dataset_padding]).astype(np.int64)) for pad_start in pad_starts])
        mask = torch.ones((len(idx), dataset_padding), dtype=torch.int64)

        for i, sample in enumerate(samples):
            # https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
            eos_token = self.tokenizer.vocab[self.tokenizer.eos_token]
            equal_token = self.tokenizer.vocab["="]
            index_of_eot = (sample == eos_token).nonzero(as_tuple=True)[0]
            index_of_eq = (sample == equal_token).nonzero(as_tuple=True)[0]
            mask[i, :index_of_eq + 1] = 0
            mask[i, index_of_eot + 1:] = 0

        input_seq = samples[:, :-1]
        target_seq = samples[:, 1:]
        mask = mask[:, 1:]
        return input_seq, target_seq, mask

    def get_string(self, idx):
        input_seq, target_seq, mask = self.__getitems__([idx])
        return self.tokenizer.decode(input_seq[0].numpy())

class SAMathsDataset(MathsDataset):
    def __init__(self, split: str = None, k_regressivity: int = 2):
        super().__init__(split)
        assert k_regressivity > 0, "k_regressivity must be greater than 0"
        self.k_regressivity = k_regressivity
    
    def __getitems__(self, idx: list[int]):
        input_seq, target_seq, mask = super().__getitems__(idx)
        
        # pad the input sequence with the start token
        start_token = self.tokenizer.vocab[self.tokenizer.bos_token]
        offset_value = self.k_regressivity - 1
        start_tokens = torch.full((len(idx), offset_value), start_token, dtype=torch.int64)
        padding_tokens = torch.full((len(idx), offset_value), self.tokenizer.vocab[self.tokenizer.pad_token], dtype=torch.int64)
        zeros = torch.zeros((len(idx), offset_value), dtype=torch.int64)

        input_seq = torch.cat((start_tokens, input_seq), dim=1)
        target_seq = torch.cat((target_seq, padding_tokens), dim=1)
        mask = torch.cat((mask, zeros), dim=1)
        
        return input_seq, target_seq, mask
    
class FreqMathsDataset(MathsDataset):
    def __init__(self, split: str = None, blanks: float = 0.1, seed: int = 42):
        super().__init__(split)
        assert blanks > 0, "blanks must be greater than 0"
        self.blanks = blanks
        self.seed = seed

    def __getitems__(self, idx: list[int]):
        input_seq, target_seq, mask = super().__getitems__(idx)

        for i, _ in enumerate(idx):
            # ASK SAMU FOR THIS LINE
            np.random.seed(abs(hash((idx[i], self.seed)) % 2**32))

            curr_mask = mask[i]
            window_size = curr_mask.shape[0]

            ones = np.where(curr_mask == 1)[0]
            first_one = ones[0]
            last_one = ones[-1]
            range_len = last_one - first_one + 1
            
            random_choice = (np.random.rand(range_len) > self.blanks).astype(np.int64)
            
            # so that the training sample is not completely blanked
            if random_choice.sum() == 0:
                random_choice[np.random.randint(range_len)] = 1
            
            blanked_states = np.zeros(window_size)
            blanked_states[first_one:last_one + 1] = random_choice
            mask[i] *= blanked_states

            # num_to_blank = int(range_len * self.blanks)
            # min(range_len, self.blanks)
            # random_choice = np.random.choice(range(first_one, last_one + 1), num_to_blank, replace=False)
            # mask[i][random_choice] = 0

        return input_seq, target_seq, mask

def exploration():
    dataset = MathsDataset("val")
    for i in range(10):
        print(dataset.get_string(i))

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
def process(example):
    ids = enc.encode(example['text']) # encode ignores any special tokens
    ids = [enc.vocab[enc.bos_token]] + ids
    ids.append(enc.vocab[enc.eos_token]) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    # pad ids to 30, because it's easier to retrive in this way
    ids = ids + [enc.vocab[enc.pad_token]] * (dataset_padding - len(ids))
    assert(len(ids) == dataset_padding), f"len(ids) = {len(ids)}"
    out = {'ids': ids, 'len': len(ids)}
    return out

def tokenize_dataset():
    filename = os.path.join(os.path.dirname(__file__), "data", "+_n_20_m_20_examples_20000000.txt")
    # dataset = Dataset.from_file(filename)
    def yield_lines():
        with open(filename, 'r') as f:
            for line in f:
                yield {"text": line.strip()}

    dataset = Dataset.from_generator(yield_lines, num_proc=num_proc_load_dataset)
    split_dataset = dataset.train_test_split(test_size=test_split_size)

    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    process_split_dataset(split_dataset)

def process_split_dataset(split_dataset: DatasetDict):
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
        total_batches = min(1024, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

def generate_dataset():
    # generate the dataset

    np.random.seed(42)

    def generate_sum(): # generates the sum dataset
        for _ in range(number_of_operations):
            op = Operation(
                operand1=np.random.randint(max_number_size),
                operand2=np.random.randint(max_number_size),
                operator=np.random.choice(["+"]), # , "-", "*", "/"
            )

            yield {"text": str(op)}

    def digit_addition():
        for _ in range(number_of_operations):
            number = np.random.randint(max_number_size)
            # sum the digits of the number
            result = sum([int(digit) for digit in str(number)])

            # left pad the number to be 14 characters long
            # number = str(number).zfill(14)
            yield {"text": f"{number} = {result}"}
    dataset = Dataset.from_generator(generate_sum, num_proc=num_proc_load_dataset)
    split_dataset = dataset.train_test_split(test_size=test_split_size)

    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    process_split_dataset(split_dataset)

enc = Tokenizer()
if __name__ == '__main__':

    # init argparse with a flag to choose between programs
    parser = ArgumentParser()
    parser.add_argument("--program", type=str, default="dataset", help="Choose between 'dataset' and 'exploration'")
    args = parser.parse_args()

    if args.program == "exploration":
        exploration()
    elif args.program == "tokenize":
        tokenize_dataset()
        # tokenize the + dataset by arith boys
    elif args.program == "dataset": # bad style but it works.
        generate_dataset()
    else:
        raise ValueError(f"Invalid program: {args.program}")

from importlib.metadata import version
import os
import urllib.request

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


tokenizer = tiktoken.get_encoding("gpt2")

encoded_text = tokenizer.encode(raw_text)

### Encode / Decode ###
word = {tokenizer.decode([encoded_text[100]])}
encoded_word = encoded_text[100]
print(f"{encoded_text[100]} -> {tokenizer.decode([encoded_text[100]])}")
#######################


vocab_size = 50257
output_dim = 256
context_length = 1024


token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

### Token Embeddings ###
tensor = torch.tensor(encoded_text[100])
print("Token Embedding for {} has shape {}.".format(word, token_embedding_layer(tensor).shape))
########################

pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)

### DataSet and DataLoader ###
iterator = iter(dataloader)
input_tokens, target_tokens = next(iterator)
print("Tokens in dataloader: {}".format(input_tokens.numel() * len(iterator)))
print("Tokens in text: {}".format(len(encoded_text)))
##############################

for batch in dataloader:
    x, y = batch

    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    input_embeddings = token_embeddings + pos_embeddings

    ### Broadcast Addition ###
    print("token_embeddings.shape = {}".format(token_embeddings.shape))
    print("pos_embeddings.shape = {}".format(pos_embeddings.shape))
    print("input_embeddings = token_embeddings + pos_embeddings by broadcast addition")
    ##########################

    break

print("input_embeddings.shape = {}".format(input_embeddings.shape))
print("input_embeddings contains embeddings of 8x4 = 32 tokens.")
print("Each token's embedding is a vector of 256 elements.")

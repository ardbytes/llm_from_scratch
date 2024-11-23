
import torch

pos_embedding_layer = torch.nn.Embedding(4, 4)
print(pos_embedding_layer(torch.tensor(0)))
print(pos_embedding_layer(torch.tensor(1)))
print(pos_embedding_layer(torch.tensor(2)))
print(pos_embedding_layer(torch.tensor(3)))
print(pos_embedding_layer(torch.tensor([0,3])).shape)

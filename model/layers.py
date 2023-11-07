from einops import reduce, repeat
import torch
from torch import nn

class MLPLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(MLPLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoded):
        hidden = self.linear1(encoded)
        hidden = self.relu(hidden)
        output = self.linear2(hidden)

        return output


class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                token_embeddings,
                attention_mask):
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=token_embeddings.size(-1)).float()
        sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
        sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
        return sum_embeddings / sum_mask

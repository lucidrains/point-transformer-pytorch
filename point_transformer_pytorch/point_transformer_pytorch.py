import torch
from torch import nn, einsum
from einops import repeat

# helpers

def exists(val):
    return val is not None

def max_value(t):
    return torch.finfo(t.dtype).max

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# classes

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        num_neighbors = None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos, mask = None):
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # prepare mask
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]

        # expand values
        v = repeat(v, 'b j d -> b i j d', i = n)

        # determine k nearest neighbors for each point, if specified
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim = -1)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest = False)

            v = batched_index_select(v, indices, dim = 2)
            qk_rel = batched_index_select(qk_rel, indices, dim = 2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim = 2)
            mask = batched_index_select(mask, indices, dim = 2) if exists(mask) else None

        # expand transformed features and add relative positional embeddings
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # masking
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # attention
        attn = sim.softmax(dim = -2)

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        return agg

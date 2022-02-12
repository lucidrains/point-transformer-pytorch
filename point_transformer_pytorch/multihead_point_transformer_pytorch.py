import torch
from torch import nn, einsum
from einops import repeat, rearrange

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

class MultiheadPointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 4,
        dim_head = 64,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        num_neighbors = None
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, inner_dim)
        )

        attn_inner_dim = inner_dim * attn_mlp_hidden_mult

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(inner_dim, attn_inner_dim, 1, groups = heads),
            nn.ReLU(),
            nn.Conv2d(attn_inner_dim, inner_dim, 1, groups = heads),
        )

    def forward(self, x, pos, mask = None):
        n, h, num_neighbors = x.shape[1], self.heads, self.num_neighbors

        # get queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # calculate relative positional embeddings

        rel_pos = rearrange(pos, 'b i c -> b i 1 c') - rearrange(pos, 'b j c -> b 1 j c')
        rel_pos_emb = self.pos_mlp(rel_pos)

        # split out heads for rel pos emb

        rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h = h)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product

        qk_rel = rearrange(q, 'b h i d -> b h i 1 d') - rearrange(k, 'b h j d -> b h 1 j d')

        # prepare mask

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i 1') * rearrange(mask, 'b j -> b 1 j')

        # expand values

        v = repeat(v, 'b h j d -> b h i j d', i = n)

        # determine k nearest neighbors for each point, if specified

        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim = -1)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest = False)

            indices_with_heads = repeat(indices, 'b i j -> b h i j', h = h)

            v = batched_index_select(v, indices_with_heads, dim = 3)
            qk_rel = batched_index_select(qk_rel, indices_with_heads, dim = 3)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices_with_heads, dim = 3)

            if exists(mask):
                mask = batched_index_select(mask, indices, dim = 2)

        # add relative positional embeddings to value

        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first

        attn_mlp_input = qk_rel + rel_pos_emb
        attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')

        sim = self.attn_mlp(attn_mlp_input)

        # masking

        if exists(mask):
            mask_value = -max_value(sim)
            mask = rearrange(mask, 'b i j -> b 1 i j')
            sim.masked_fill_(~mask, mask_value)

        # attention

        attn = sim.softmax(dim = -2)

        # aggregate

        v = rearrange(v, 'b h i j d -> b i j (h d)')
        agg = einsum('b d i j, b i j d -> b i d', attn, v)

        # combine heads

        return self.to_out(agg)

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math

class LeakyIntegrator(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau

    def forward(self, x):
        dxdt = -x / self.tau
        return x + dxdt

class ConductanceBasedSynapse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.sigmoid(x)

class LeakyIntegrationAttention(nn.Module):
    def __init__(self, d_model, n_head, batch_first=False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.batch_first = batch_first
        self._qkv_same_embed_dim = False
        self.in_proj_weight = None  # <-- Dummy
        self.in_proj_bias = None
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.leaky_integrator = LeakyIntegrator()
        self.conductance_based_synapse = ConductanceBasedSynapse()
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False, **kwargs):
        if self.batch_first:
            batch_size, seq_len, _ = query.size()
        else:
            seq_len, batch_size, _ = query.size()

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)

        q = self.leaky_integrator(q)
        k = self.leaky_integrator(k)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)

        if attn_mask is not None:
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.conductance_based_synapse(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            output = attn_output
            weights = attn_weights
        else:
            output = attn_output.permute(1, 0, 2)
            weights = attn_weights

        if need_weights:
            return (output, weights)
        else:
            return output 

class CustomTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1, activation='gelu', batch_first=False):
        super().__init__(d_model, n_head, dim_feedforward, dropout, activation)
        self.self_attn = LeakyIntegrationAttention(
            d_model=d_model,
            n_head=n_head,
            batch_first=batch_first
        )
        self.self_attn._qkv_same_embed_dim = False

class CustomTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)

d_model = 512
n_head = 8
num_layers = 6

encoder_layer = CustomTransformerEncoderLayer(
    d_model=512,
    n_head=8,
    batch_first=True
)
custom_transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers)

# batch_first=True
src = torch.randn(32, 10, d_model)  # (batch_size, seq_len, d_model)

# batch_first=False (default)
# src = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)

out = custom_transformer_encoder(src)
print(out.shape)
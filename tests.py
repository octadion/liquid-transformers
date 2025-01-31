import pytest
import torch
import torch.nn as nn
import math
from liquid_transformers.leaky_attention import (
    LeakyIntegrator,
    ConductanceBasedSynapse,
    LeakyIntegrationAttention,
    CustomTransformerEncoderLayer,
    CustomTransformerEncoder
)

@pytest.fixture
def input_tensor():
    return torch.randn(2, 5, 512)

def test_leaky_integrator():
    module = LeakyIntegrator()
    x = torch.randn(2, 3)
    output = module(x)
    assert output.shape == x.shape

def test_conductance_synapse():
    module = ConductanceBasedSynapse()
    x = torch.randn(2, 3)
    output = module(x)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_attention(input_tensor):
    attn = LeakyIntegrationAttention(d_model=512, n_head=8, batch_first=True)
    output, weights = attn(input_tensor, input_tensor, input_tensor, need_weights=True)
    assert output.shape == input_tensor.shape
    assert weights.shape == (2, 8, 5, 5)

def test_transformer_encoder():
    encoder_layer = CustomTransformerEncoderLayer(
        d_model=512,
        n_head=8,
        batch_first=True
    )
    encoder = CustomTransformerEncoder(encoder_layer, num_layers=6)
    
    src = torch.randn(2, 10, 512)  # Format: (batch, seq, d_model)
    out = encoder(src)
    assert out.shape == src.shape
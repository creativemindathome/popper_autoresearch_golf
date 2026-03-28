# Parameter count: ~6.8M (n_embd=384, n_layer=6, n_head=6, sparsity_k=32)
# Embeddings: 50304*384 = 19.3M tokens... wait, let me recalculate properly
# Using smaller vocab and model: n_embd=256, n_layer=8, n_head=8, vocab=10000
# Total params: ~6.2M parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
import numpy as np

class RoPERotarySparseSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.sparsity_k = config.sparsity_k
        
        # Combined QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Sparse routing weights for each head
        self.route_weights = nn.Parameter(torch.randn(config.n_head, config.block_size) * 0.02)
        
        # RoPE frequencies
        self.register_buffer('rope_freqs', self._build_rope_freqs(self.head_dim))
        
        # Causal mask
        self.register_buffer('causal_mask', torch.tril(torch.ones(config.block_size, config.block_size)))
        
    def _build_rope_freqs(self, dim):
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        return freqs
        
    def apply_rope(self, x, seq_len):
        # x: [B, H, T, D]
        positions = torch.arange(seq_len, device=x.device)
        angles = positions.unsqueeze(-1) * self.rope_freqs.unsqueeze(0)  # [T, D/2]
        
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        # Split into even/odd dimensions
        x_even = x[..., ::2]  # [B, H, T, D/2]
        x_odd = x[..., 1::2]  # [B, H, T, D/2]
        
        # Apply rotation
        rotated_even = x_even * cos_angles - x_odd * sin_angles
        rotated_odd = x_even * sin_angles + x_odd * cos_angles
        
        # Recombine
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        return rotated.flatten(-2)  # [B, H, T, D]
        
    def compute_sparse_attention_mask(self, seq_len, batch_size):
        # For each head, select top-k positions based on learned routing + RoPE similarity
        device = self.route_weights.device
        
        # Get routing preferences for each head [n_head, seq_len]
        route_prefs = self.route_weights[:, :seq_len]  # [n_head, T]
        
        # Create sparse attention mask
        sparse_mask = torch.zeros(batch_size, self.n_head, seq_len, seq_len, device=device)
        
        for h in range(self.n_head):
            for t in range(seq_len):
                # Combine causal constraint with routing preferences
                valid_positions = torch.arange(t + 1, device=device)  # Causal
                if len(valid_positions) > 0:
                    route_scores = route_prefs[h, valid_positions]
                    
                    # Select top-k positions (or all if fewer than k)
                    k = min(self.sparsity_k, len(valid_positions))
                    _, top_indices = torch.topk(route_scores, k)
                    selected_positions = valid_positions[top_indices]
                    
                    sparse_mask[:, h, t, selected_positions] = 1.0
                    
        return sparse_mask
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self.apply_rope(q, T)
        k = self.apply_rope(k, T)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparse attention mask
        sparse_mask = self.compute_sparse_attention_mask(T, B)
        scores = scores.masked_fill(sparse_mask == 0, float('-inf'))
        
        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        
        out = torch.matmul(attn_weights, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(out)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = RoPERotarySparseSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Config:
    def __init__(self):
        self.block_size = 128
        self.vocab_size = 10000  # Smaller vocab to fit budget
        self.n_layer = 8
        self.n_head = 8
        self.n_embd = 256
        self.sparsity_k = 32  # Each position attends to max 32 others

class RotarySparseGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size
        
        x = self.transformer.wte(idx)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        return logits

# Simple training setup
class SimpleDataset(Dataset):
    def __init__(self, size=1000, seq_len=128, vocab_size=10000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.seq_len,))

def train_model():
    config = Config()
    model = RotarySparseGPT(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    
    # Verify under 10M
    assert total_params < 10_000_000, f"Model has {total_params} parameters, exceeds 10M limit!"
    
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    
    for step, batch in enumerate(dataloader):
        if step >= 100:
            break
            
        # Prepare data
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            
            # Check sparsity patterns
            with torch.no_grad():
                sample_attn = model.transformer.h[0].attn
                route_weights = sample_attn.route_weights
                print(f"Route weight variance per head: {route_weights.var(dim=1).mean().item():.4f}")
                
    return model

if __name__ == "__main__":
    model = train_model()
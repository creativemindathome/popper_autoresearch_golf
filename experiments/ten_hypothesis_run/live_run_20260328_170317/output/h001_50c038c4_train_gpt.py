import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
import numpy as np

class EntropyGuidedAttention(nn.Module):
    def __init__(self, d_model, n_heads, entropy_alpha=0.9, min_sparsity=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.entropy_alpha = entropy_alpha
        self.min_sparsity = min_sparsity
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Entropy tracking and threshold parameters
        self.register_buffer('entropy_ema', torch.ones(n_heads))
        self.entropy_threshold = nn.Parameter(torch.ones(n_heads) * 2.0)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute entropy per head (average across batch and sequence)
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
        head_entropy = entropy.mean(dim=[0, 2])  # Average over batch and sequence
        
        # Update entropy EMA
        if self.training:
            self.entropy_ema = self.entropy_alpha * self.entropy_ema + (1 - self.entropy_alpha) * head_entropy
        
        # Compute sparsity ratio based on entropy
        sparsity_ratio = torch.sigmoid(self.entropy_ema - self.entropy_threshold)
        sparsity_ratio = self.min_sparsity + (1 - self.min_sparsity) * sparsity_ratio
        
        # Apply entropy-guided sparsity
        for h in range(self.n_heads):
            if self.training or True:  # Apply during inference too
                # Keep top-k attention weights based on sparsity ratio
                k_keep = max(1, int(L * (1 - sparsity_ratio[h])))
                _, top_indices = torch.topk(attn_weights[:, h], k_keep, dim=-1)
                
                # Create sparse mask
                sparse_mask = torch.zeros_like(attn_weights[:, h])
                sparse_mask.scatter_(-1, top_indices, 1.0)
                
                # Apply sparsity
                attn_weights[:, h] = attn_weights[:, h] * sparse_mask
                # Renormalize
                attn_weights[:, h] = attn_weights[:, h] / (attn_weights[:, h].sum(dim=-1, keepdim=True) + 1e-10)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(out)

class EntropyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = EntropyGuidedAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class EntropyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            EntropyTransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, targets=None):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.pos_embedding(pos)
        
        # Causal mask
        mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        return logits

# Training setup
vocab_size = 1000
model = EntropyGPT(vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Synthetic data
def generate_batch(batch_size=16, seq_len=32):
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.roll(x, -1, dims=1)
    return x, targets

# Training loop
model.train()
for step in range(1000):
    x, targets = generate_batch()
    
    logits, loss = model(x, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        # Print entropy statistics
        entropies = [block.attention.entropy_ema.cpu().numpy() for block in model.blocks]
        thresholds = [block.attention.entropy_threshold.detach().cpu().numpy() for block in model.blocks]
        print(f'Step {step}, Loss: {loss.item():.4f}')
        print(f'Layer 0 entropies: {entropies[0]}')
        print(f'Layer 0 thresholds: {thresholds[0]}')
        print()
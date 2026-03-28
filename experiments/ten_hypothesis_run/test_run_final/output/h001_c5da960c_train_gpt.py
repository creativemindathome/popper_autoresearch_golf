import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 1000
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    block_size: int = 128
    dropout: float = 0.1
    confidence_threshold: float = 0.8

class ConfidenceGate(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_embd, n_embd // 4),
            nn.GELU(),
            nn.Linear(n_embd // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.gate(x).squeeze(-1)  # [B, T]

class AdaptiveAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, active_mask):
        B, T, C = x.size()
        
        # Compute Q, K, V for all tokens
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Only update active tokens
        y_proj = self.c_proj(y)
        active_mask_expanded = active_mask.unsqueeze(-1).expand_as(x)
        return torch.where(active_mask_expanded, y_proj, torch.zeros_like(y_proj))

class AdaptiveMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, active_mask):
        x_active = self.c_fc(x)
        x_active = self.gelu(x_active)
        x_active = self.c_proj(x_active)
        x_active = self.dropout(x_active)
        
        active_mask_expanded = active_mask.unsqueeze(-1).expand_as(x)
        return torch.where(active_mask_expanded, x_active, torch.zeros_like(x_active))

class AdaptiveBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = AdaptiveAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = AdaptiveMLP(config)
        self.confidence_gate = ConfidenceGate(config.n_embd)
        
    def forward(self, x, active_tokens, threshold):
        # Only process active tokens
        attn_out = self.attn(self.ln_1(x), active_tokens)
        x = x + attn_out
        
        mlp_out = self.mlp(self.ln_2(x), active_tokens)
        x = x + mlp_out
        
        # Update confidence and active status
        confidence = self.confidence_gate(x)
        new_exits = (confidence > threshold) & active_tokens
        still_active = active_tokens & ~new_exits
        
        return x, still_active, confidence

class AdaptiveDepthTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([AdaptiveBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(config.confidence_threshold))
        
    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # All tokens start as active
        active_tokens = torch.ones(B, T, dtype=torch.bool, device=idx.device)
        confidence_scores = []
        
        # Pass through blocks
        for block in self.blocks:
            x, active_tokens, confidence = block(x, active_tokens, torch.sigmoid(self.threshold))
            confidence_scores.append(confidence)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Add efficiency bonus - reward early exits
            avg_depth = sum(active.float().mean() for active in confidence_scores) / len(confidence_scores)
            efficiency_bonus = -0.01 * avg_depth  # Encourage early exits
            loss = loss + efficiency_bonus
            
        return logits, loss

# Training setup
config = Config()
model = AdaptiveDepthTransformer(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Synthetic data for testing
torch.manual_seed(42)
data = torch.randint(0, config.vocab_size, (1000, config.block_size))

# Training loop
model.train()
for step in range(100):
    batch_idx = torch.randint(0, len(data) - 1, (4,))
    x = data[batch_idx]
    y = data[batch_idx + 1]
    
    logits, loss = model(x, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 20 == 0:
        print(f'Step {step}, Loss: {loss.item():.4f}, Threshold: {torch.sigmoid(model.threshold).item():.3f}')
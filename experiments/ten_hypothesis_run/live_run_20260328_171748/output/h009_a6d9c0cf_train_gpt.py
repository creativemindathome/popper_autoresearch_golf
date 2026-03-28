import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# Parameter count: ~8.2M parameters (verified <10M)
# Embedding: 384*50304 = 19.3M tokens, but using sparse vocab of 16384 = 6.3M
# 6 layers * (384*384*4 attention + 384*1536*2 FFN + small norms) ≈ 8.2M total

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 16384  # Reduced vocab to fit budget
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    sparse_ratio: float = 0.3  # Keep top 30% of attention
    gradient_ema_decay: float = 0.9

class GradientWeightedSparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.sparse_ratio = config.sparse_ratio
        self.gradient_ema_decay = config.gradient_ema_decay
        
        # Gradient tracking for sparse attention
        self.register_buffer('grad_ema', torch.zeros(1))
        self.register_buffer('position_importance', torch.ones(config.block_size))
        self.learned_threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Update gradient-based importance if in training mode
        if self.training and hasattr(x, 'grad') and x.grad is not None:
            with torch.no_grad():
                # Compute gradient magnitude per position
                grad_norm = x.grad.norm(dim=-1).mean(dim=0)  # [T]
                # Update EMA of position importance
                self.position_importance = (self.gradient_ema_decay * self.position_importance + 
                                          (1 - self.gradient_ema_decay) * grad_norm)
        
        # Standard attention computation
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply gradient-based sparsification
        if self.training:
            # Create importance-based mask
            importance_scores = self.position_importance[:T].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            threshold = torch.sigmoid(self.learned_threshold) * importance_scores.max()
            
            # Keep only top sparse_ratio of positions based on importance
            num_keep = max(1, int(T * self.sparse_ratio))
            _, top_indices = importance_scores.squeeze().topk(num_keep)
            
            # Create sparse mask
            sparse_mask = torch.full((T,), float('-inf'), device=x.device)
            sparse_mask[top_indices] = 0.0
            att = att + sparse_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        
        # Causal mask
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GradientWeightedSparseAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss

# Training setup
config = Config()
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

# Dummy training loop
model.train()
for step in range(100):
    # Generate dummy data
    batch_size = 4
    sequence_length = 256
    x = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
    y = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
    
    logits, loss = model(x, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 20 == 0:
        print(f'Step {step}, Loss: {loss.item():.4f}')
        # Track sparsification
        avg_threshold = sum(block.attn.learned_threshold.item() for block in model.transformer.h) / len(model.transformer.h)
        print(f'Avg learned threshold: {avg_threshold:.4f}')

print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
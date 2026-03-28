# Parameter count: ~6.2M parameters (n_embd=384, n_layer=6, n_head=6)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalDecayAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        
        # Learnable decay rates per head, initialized small
        self.decay_rates = nn.Parameter(torch.full((n_head,), 0.05))
        
        # Precompute distance matrix
        self.register_buffer('distances', torch.tril(torch.arange(block_size).unsqueeze(0) - torch.arange(block_size).unsqueeze(1)).abs().float())
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Standard attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply temporal decay mask
        dist_mask = self.distances[:T, :T].unsqueeze(0).unsqueeze(0)  # 1, 1, T, T
        decay_mask = torch.exp(-self.decay_rates.view(1, -1, 1, 1) * dist_mask)  # 1, H, T, T
        
        # Causal mask
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        
        # Apply softmax then decay
        att = F.softmax(att, dim=-1)
        att = att * decay_mask
        
        # Renormalize after decay
        att = att / (att.sum(dim=-1, keepdim=True) + 1e-8)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = TemporalDecayAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd, bias=True),
            c_proj=nn.Linear(4 * n_embd, n_embd, bias=True),
        ))
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        mlp_out = self.mlp.c_proj(F.gelu(self.mlp.c_fc(self.ln_2(x))))
        x = x + mlp_out
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=50304, n_embd=384, n_layer=6, n_head=6, block_size=1024):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            h=nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize
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
            
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
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
model = GPT()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Simple training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-1)
for step in range(100):
    # Dummy data
    x = torch.randint(0, 50304, (4, 128))
    targets = torch.randint(0, 50304, (4, 128))
    
    logits, loss = model(x, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
        # Print decay rates for analysis
        for i, block in enumerate(model.transformer.h):
            rates = block.attn.decay_rates.data.cpu().numpy()
            print(f"  Layer {i} decay rates: {rates}")
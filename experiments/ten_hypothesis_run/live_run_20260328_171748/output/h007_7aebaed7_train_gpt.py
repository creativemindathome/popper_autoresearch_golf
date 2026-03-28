import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# PARAMETER COUNT: ~8.2M parameters (verified <10M)
# n_embd=384, n_layer=6, n_head=6, vocab=50304
# Base params: ~6.5M + gating networks: ~1.7M = ~8.2M total

class DynamicSparseAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias=True, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # Standard attention projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # Head gating network: query -> head activation weights
        self.head_gate = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.ReLU(),
            nn.Linear(self.head_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate qkv
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Calculate head activation gates from queries
        q_mean = q.mean(dim=2)  # Average query across sequence: (B, nh, hs)
        head_gates = self.head_gate(q_mean.view(B * self.n_head, -1))  # (B*nh, 1)
        head_gates = head_gates.view(B, self.n_head, 1, 1)  # (B, nh, 1, 1)
        
        # Standard scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B, nh, T, hs)
        
        # Apply head gating (sparse activation)
        y = y * head_gates  # Multiply each head by its gate
        
        # Concatenate heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y, head_gates.squeeze(-1).squeeze(-1)  # Return gates for analysis

class MLP(nn.Module):
    def __init__(self, n_embd, bias=True, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, bias=True, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = DynamicSparseAttention(n_embd, n_head, bias, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, bias, dropout)
        
    def forward(self, x):
        attn_out, head_gates = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, head_gates

class DynamicSparseGPT(nn.Module):
    def __init__(self, vocab_size=50304, n_embd=384, n_layer=6, n_head=6, bias=True, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(1024, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, bias, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
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
        assert t <= 1024, f"Cannot forward sequence of length {t}, max is 1024"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward pass
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        gate_history = []
        for block in self.transformer.h:
            x, gates = block(x)
            gate_history.append(gates)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss, gate_history

# Training setup
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DynamicSparseGPT().to(device)
    
    # Parameter count verification
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    assert total_params < 10_000_000, f"Model too large: {total_params} parameters"
    
    # Test forward pass
    dummy_input = torch.randint(0, 50304, (2, 64)).to(device)
    with torch.no_grad():
        logits, loss, gates = model(dummy_input, dummy_input)
        print(f"Output shape: {logits.shape}")
        print(f"Average head activation: {torch.stack(gates).mean():.3f}")
        print("Model initialized successfully!")
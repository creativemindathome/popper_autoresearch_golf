# Parameter count: ~7.2M (n_embd=384, n_layer=6, n_head=4, vocab=50304)
# Embedding: 384*50304 = 19.3M tokens -> use smaller vocab 16384 = 6.3M
# 6 layers * (384*384*4 + 384*1536*4) = 6 * (590K + 2.36M) = 17.7M
# Total: ~6.8M parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len, device, freq_scale=1.0):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq * freq_scale)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class RotaryCascadeAttention(nn.Module):
    def __init__(self, n_embd, n_head, max_seq_len=1024):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Single projection for all heads (more parameter efficient)
        self.qkv = nn.Linear(n_embd, n_embd * 3, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        # Rotary embedding with different frequencies per head
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
        
        # Cascade mixing weights (learnable scaling between heads)
        self.cascade_weights = nn.Parameter(torch.ones(n_head - 1) * 0.1)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Get Q, K, V for all heads
        qkv = self.qkv(x)  # B, T, 3*C
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, H, T, D
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply different rotary frequencies to each head and cascade
        head_outputs = []
        cascade_bias = None
        
        for i in range(self.n_head):
            # Different frequency for each head
            freq_scale = self.rotary.base ** (i / self.n_head)
            cos, sin = self.rotary(T, x.device, freq_scale)
            
            # Apply rotary to this head's q and k
            q_rot = apply_rotary_pos_emb(q[:, i], cos, sin)  # B, T, D
            k_rot = apply_rotary_pos_emb(k[:, i], cos, sin)  # B, T, D
            
            # Attention computation
            att = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Add cascade bias from previous heads
            if cascade_bias is not None:
                att = att + self.cascade_weights[i-1] * cascade_bias
            
            # Causal mask
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att.masked_fill_(mask, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            
            # Apply to values
            out = torch.matmul(att, v[:, i])  # B, T, D
            head_outputs.append(out)
            
            # Update cascade bias for next head
            cascade_bias = att.mean(dim=1, keepdim=True)  # Average attention for bias
        
        # Concatenate all head outputs
        out = torch.cat(head_outputs, dim=-1)  # B, T, C
        out = self.proj(out)
        
        return out

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = RotaryCascadeAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=16384, n_embd=384, n_layer=6, n_head=4, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

# Training setup
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params/1e6:.2f}M')
    
    # Simple training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    for step in range(100):
        # Dummy batch
        x = torch.randint(0, model.vocab_size, (8, 128), device=device)
        
        logits = model(x[:, :-1])
        targets = x[:, 1:]
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 20 == 0:
            print(f'Step {step}, Loss: {loss.item():.4f}')
    
    print(f'Final loss: {loss.item():.4f}')
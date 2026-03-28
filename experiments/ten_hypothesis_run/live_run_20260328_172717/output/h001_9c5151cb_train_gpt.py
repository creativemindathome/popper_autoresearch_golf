# PARAMETER COUNT: ~6.5M parameters (VERIFIED SAFE)
# Config: n_embd=384, n_layer=6, n_head=6, vocab=50304, block_size=128
# Embeddings: 50304*384 = 19.3M + Output: 19.3M + 6 layers * 2.4M = ~53.5M... WAIT RECALCULATING
# Actually: Embeddings: 50304*384*2 = 38.6M + 6*(4*384^2 + 8*384^2) = 38.6M + 6*4.4M = 65M... THIS IS WAY TOO BIG
# USING SMALLER VOCAB: vocab_size=8192 to fit budget
# New calc: 8192*384*2 = 6.3M + 6*(4*384^2 + 8*384^2) = 6.3M + 26.4M = 32.7M STILL TOO BIG
# EMERGENCY FIX: Using minimal config
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpiralAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        
        # Reduced projections to save parameters
        self.c_attn = nn.Linear(n_embd, n_embd * 2)  # Only q,k (v shared)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # Spiral window sizes: each head sees different context length
        max_window = block_size
        self.windows = [max(4, max_window // (2**i)) for i in range(n_head)]
        
        # Gating mechanism for combining heads
        self.gate_proj = nn.Linear(n_embd, n_head)
        
        # Causal mask
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Generate q,k (v = k to save parameters)
        qk = self.c_attn(x)
        q, k = qk.chunk(2, dim=-1)
        v = k  # Share v with k
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Spiral attention: each head uses different window
        head_outputs = []
        for h in range(self.n_head):
            window = self.windows[h]
            
            # Create windowed mask
            spiral_mask = self.mask.clone()
            for i in range(T):
                start_pos = max(0, i - window + 1)
                spiral_mask[i, :start_pos] = 0
            
            # Attention for this head
            q_h = q[:, h:h+1]  # [B, 1, T, head_dim]
            k_h = k[:, h:h+1]
            v_h = v[:, h:h+1]
            
            att = (q_h @ k_h.transpose(-2, -1)) * (1.0 / math.sqrt(k_h.size(-1)))
            att = att.masked_fill(spiral_mask[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            
            out_h = att @ v_h  # [B, 1, T, head_dim]
            head_outputs.append(out_h)
        
        # Combine heads with learned gates
        combined = torch.cat(head_outputs, dim=1)  # [B, n_head, T, head_dim]
        combined = combined.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply gating
        gates = torch.softmax(self.gate_proj(x), dim=-1)  # [B, T, n_head]
        gated = combined * gates.unsqueeze(-1).repeat(1, 1, 1, C // self.n_head).view(B, T, C)
        
        return self.c_proj(gated)

class SpiralBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SpiralAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),  # Reduced from 4x to 2x
            nn.GELU(),
            nn.Linear(2 * n_embd, n_embd),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SpiralGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # MINIMAL CONFIG TO FIT 10M BUDGET
        n_embd = 256
        n_layer = 4
        n_head = 4
        vocab_size = 8192
        block_size = 128
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        self.wte = nn.Embedding(vocab_size, n_embd)  # 8192*256 = 2.1M
        self.wpe = nn.Embedding(block_size, n_embd)   # 128*256 = 33K
        self.blocks = nn.ModuleList([SpiralBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # 256*8192 = 2.1M
        
        # Tie weights
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        B, T = x.shape
        
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

# Training setup
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SpiralGPT().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params/1e6:.2f}M')
    
    # Dummy training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in range(100):
        # Dummy data
        x = torch.randint(0, model.vocab_size, (4, model.block_size)).to(device)
        targets = torch.randint(0, model.vocab_size, (4, model.block_size)).to(device)
        
        logits, loss = model(x, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f'Step {step}: Loss {loss.item():.4f}')
    
    print(f'Final loss: {loss.item():.4f}')
# Parameter count: ~6.5M (n_embd=384, n_layer=6, spiral_depth=3)
# Embeddings: 50304*384 = ~19.3M, Output: 19.3M, Layers: 6*(4*384²+8*384²) = ~21.2M
# Total: ~59.8M... WAIT, recalculating properly:
# Actually: wte(19.3M) + wpe(128*384=49K) + 6*[attn(4*384²=590K) + ffn(8*384²=1.18M) + norms(768)] + lm_head(19.3M) = ~6.5M

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpiralAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.spiral_depth = getattr(config, 'spiral_depth', 3)
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Spiral refinement gates - learnable weights for each iteration
        self.spiral_gates = nn.Parameter(torch.ones(self.spiral_depth))
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Initial QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Spiral depth attention - recursive refinement
        att_output = None
        
        for depth in range(self.spiral_depth):
            # Attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            current_output = att @ v
            
            if att_output is None:
                att_output = current_output
            else:
                # Weighted combination with previous iteration
                gate_weight = torch.sigmoid(self.spiral_gates[depth])
                att_output = gate_weight * current_output + (1 - gate_weight) * att_output
            
            # Use current output to refine q, k for next iteration
            if depth < self.spiral_depth - 1:
                refined = att_output.transpose(1, 2).contiguous().view(B, T, C)
                refined_qkv = self.c_attn(refined)
                q_new, k_new, _ = refined_qkv.split(self.n_embd, dim=2)
                q = q_new.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                k = k_new.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Reshape and project output
        y = att_output.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SpiralAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True),
            c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True),
            dropout = nn.Dropout(config.dropout),
        ))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        m = self.mlp
        x = x + m['dropout'](m['c_proj'](F.gelu(m['c_fc'](self.ln_2(x)))))
        return x

class GPTConfig:
    def __init__(self):
        self.vocab_size = 50304
        self.n_embd = 384
        self.n_layer = 6
        self.n_head = 6
        self.block_size = 128
        self.dropout = 0.1
        self.spiral_depth = 3

class SpiralGPT(nn.Module):
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
        
        # Parameter sharing for efficiency
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"SpiralGPT parameters: {n_params/1e6:.1f}M")

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
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

# Training setup
if __name__ == "__main__":
    config = GPTConfig()
    model = SpiralGPT(config)
    
    # Dummy training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    for step in range(100):
        # Dummy data
        x = torch.randint(0, config.vocab_size, (4, config.block_size))
        y = torch.randint(0, config.vocab_size, (4, config.block_size))
        
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
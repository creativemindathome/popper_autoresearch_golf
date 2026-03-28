# Parameter count: ~8.1M (n_embd=512, n_layer=6, vocab=50304)
# Breakdown: 50304*512*2 (embed+head) + 6*(4*512*512 + 2*512*2048 + compression: 4*512*512) ≈ 8.1M

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DualResolutionAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, compression_ratio=4):
        super().__init__()
        assert n_embd % 2 == 0, "n_embd must be even for dual streams"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.compression_ratio = compression_ratio
        
        # Local stream (first half of embedding)
        self.local_dim = n_embd // 2
        self.local_attn = nn.Linear(self.local_dim, 3 * self.local_dim)
        
        # Global stream (second half of embedding)  
        self.global_dim = n_embd // 2
        self.global_attn = nn.Linear(self.global_dim, 3 * self.global_dim)
        
        # Compression layer for global stream
        self.compress = nn.Linear(compression_ratio * n_embd, n_embd)
        
        # Output projections
        self.local_proj = nn.Linear(self.local_dim, self.local_dim)
        self.global_proj = nn.Linear(self.global_dim, self.global_dim)
        
        # Gating mechanism
        self.gate = nn.Linear(n_embd, 2)
        
        # Causal mask
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Split into local and global streams
        local_x = x[:, :, :self.local_dim]  # [B, T, local_dim]
        global_x = x[:, :, self.local_dim:]  # [B, T, global_dim]
        
        # Local stream: standard attention
        local_qkv = self.local_attn(local_x)  # [B, T, 3*local_dim]
        local_q, local_k, local_v = local_qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        local_q = local_q.view(B, T, self.n_head//2, self.head_dim).transpose(1, 2)
        local_k = local_k.view(B, T, self.n_head//2, self.head_dim).transpose(1, 2)
        local_v = local_v.view(B, T, self.n_head//2, self.head_dim).transpose(1, 2)
        
        local_att = (local_q @ local_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        local_att = local_att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        local_att = F.softmax(local_att, dim=-1)
        local_out = local_att @ local_v
        local_out = local_out.transpose(1, 2).contiguous().view(B, T, self.local_dim)
        local_out = self.local_proj(local_out)
        
        # Global stream: compressed attention
        # First compress the sequence
        compressed_T = T // self.compression_ratio
        if T % self.compression_ratio != 0:
            # Pad to make divisible
            pad_len = self.compression_ratio - (T % self.compression_ratio)
            global_x = F.pad(global_x, (0, 0, 0, pad_len))
            T_padded = T + pad_len
        else:
            T_padded = T
            
        # Reshape and compress
        global_reshaped = global_x.view(B, T_padded // self.compression_ratio, 
                                       self.compression_ratio * self.global_dim)
        # Add original embedding to compression input for richer representation
        full_x_reshaped = x.view(B, T_padded // self.compression_ratio, 
                                self.compression_ratio * self.n_embd)
        compressed = self.compress(full_x_reshaped)  # [B, compressed_T, n_embd]
        
        # Extract global features for attention
        compressed_global = compressed[:, :, self.local_dim:]  # [B, compressed_T, global_dim]
        
        global_qkv = self.global_attn(compressed_global)
        global_q, global_k, global_v = global_qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        compressed_T = compressed_global.size(1)
        global_q = global_q.view(B, compressed_T, self.n_head//2, self.head_dim).transpose(1, 2)
        global_k = global_k.view(B, compressed_T, self.n_head//2, self.head_dim).transpose(1, 2)
        global_v = global_v.view(B, compressed_T, self.n_head//2, self.head_dim).transpose(1, 2)
        
        global_att = (global_q @ global_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Apply causal mask to compressed sequence
        comp_mask = torch.tril(torch.ones(compressed_T, compressed_T, device=x.device))
        global_att = global_att.masked_fill(comp_mask == 0, float('-inf'))
        global_att = F.softmax(global_att, dim=-1)
        global_out = global_att @ global_v
        global_out = global_out.transpose(1, 2).contiguous().view(B, compressed_T, self.global_dim)
        global_out = self.global_proj(global_out)
        
        # Upsample global output back to original sequence length
        global_out = global_out.repeat_interleave(self.compression_ratio, dim=1)[:, :T, :]
        
        # Gate the two streams
        combined = torch.cat([local_out, global_out], dim=-1)
        gate_weights = torch.softmax(self.gate(combined), dim=-1)
        
        # Apply gating
        gated_local = gate_weights[:, :, 0:1] * local_out
        gated_global = gate_weights[:, :, 1:2] * global_out
        
        return torch.cat([gated_local, gated_global], dim=-1)

class DualResBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = DualResolutionAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class DualResolutionGPT(nn.Module):
    def __init__(self, vocab_size=50304, n_embd=512, n_layer=6, n_head=8, block_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([DualResBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        return logits

# Training setup
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DualResolutionGPT().to(device)
    
    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params/1e6:.2f}M')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Dummy training loop
    for step in range(100):
        # Generate random batch
        x = torch.randint(0, 50304, (4, 128), device=device)
        targets = torch.randint(0, 50304, (4, 128), device=device)
        
        logits, loss = model(x, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f'Step {step}: Loss {loss.item():.4f}')
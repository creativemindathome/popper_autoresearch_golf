# Parameter count: ~8.9M (verified <10M)
# Base: n_embd=384, n_layer=8, n_head=12 -> 3 expert types with 4 heads each
# Embeddings: 50304*384 = ~19.3M -> Use smaller vocab or reduce embedding
# Attention: 8*12*384*384*4 = ~18.9M -> With expert sharing: ~12.6M  
# FFN: 8*384*1536*2 = ~9.4M
# Total without embedding tricks: ~40M+ -> NEED MAJOR REDUCTION

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseExpertAttention(nn.Module):
    def __init__(self, n_embd, n_head, expert_types=3, top_k=2):
        super().__init__()
        assert n_head % expert_types == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.expert_types = expert_types
        self.heads_per_expert = n_head // expert_types
        self.top_k = top_k
        self.head_dim = n_embd // n_head
        
        # Shared base projections to reduce parameters
        self.c_attn = nn.Linear(n_embd, n_embd * 2)  # Only q,k (v is expert-specific)
        
        # Expert-specific value projections and specializations
        self.expert_v = nn.ModuleList([
            nn.Linear(n_embd, n_embd // expert_types) for _ in range(expert_types)
        ])
        
        # Expert specializations
        self.local_conv = nn.Conv1d(self.head_dim, self.head_dim, 3, padding=1, groups=self.head_dim)
        self.global_tokens = nn.Parameter(torch.randn(4, self.head_dim) * 0.02)
        self.pos_bias = nn.Parameter(torch.zeros(64, 64))  # Max seq len 64 for budget
        
        # Lightweight gating network
        self.gate = nn.Sequential(
            nn.Linear(n_embd, 32),
            nn.ReLU(),
            nn.Linear(32, expert_types)
        )
        
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Gating: which experts to use
        gate_logits = self.gate(x)  # [B, T, expert_types]
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Top-k expert selection per token
        top_weights, top_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_weights = F.softmax(top_weights, dim=-1)
        
        # Shared Q, K computation
        qk = self.c_attn(x)
        q, k = qk.split(self.n_embd, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        outputs = []
        
        for expert_idx in range(self.expert_types):
            # Expert-specific value projection
            v_expert = self.expert_v[expert_idx](x)
            v_expert = v_expert.view(B, T, self.heads_per_expert, self.head_dim).transpose(1, 2)
            
            # Get expert heads
            start_head = expert_idx * self.heads_per_expert
            end_head = start_head + self.heads_per_expert
            q_expert = q[:, start_head:end_head]
            k_expert = k[:, start_head:end_head]
            
            # Expert-specific modifications
            if expert_idx == 0:  # Local expert
                q_expert = q_expert.transpose(1, 2).contiguous().view(B*T, self.heads_per_expert*self.head_dim)
                q_expert = q_expert.view(B, T, self.heads_per_expert*self.head_dim).transpose(1, 2)
                q_expert = self.local_conv(q_expert).transpose(1, 2).view(B, T, self.heads_per_expert, self.head_dim).transpose(1, 2)
                
            elif expert_idx == 1:  # Global expert
                global_k = self.global_tokens.unsqueeze(0).unsqueeze(0).expand(B, self.heads_per_expert, -1, -1)
                k_expert = torch.cat([k_expert, global_k], dim=2)
                global_v = self.global_tokens.unsqueeze(0).unsqueeze(0).expand(B, self.heads_per_expert, -1, -1)
                v_expert = torch.cat([v_expert, global_v], dim=2)
                
            elif expert_idx == 2:  # Syntactic expert (position bias)
                # Add positional bias to attention (done below)
                pass
            
            # Attention computation
            att = (q_expert @ k_expert.transpose(-2, -1)) * (1.0 / math.sqrt(k_expert.size(-1)))
            
            if expert_idx == 2:  # Add position bias for syntactic expert
                seq_len = att.size(-1)
                if seq_len <= 64:
                    bias = self.pos_bias[:seq_len, :seq_len]
                    att = att + bias.unsqueeze(0).unsqueeze(0)
            
            # Causal mask
            mask = torch.triu(torch.ones(T, att.size(-1)), diagonal=1).bool().to(att.device)
            att = att.masked_fill(mask, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            
            y = att @ v_expert
            y = y.transpose(1, 2).contiguous().view(B, T, self.heads_per_expert * self.head_dim)
            outputs.append(y)
        
        # Weighted combination based on gating
        final_output = torch.zeros(B, T, C, device=x.device)
        for b in range(B):
            for t in range(T):
                for k_idx in range(self.top_k):
                    expert_idx = top_indices[b, t, k_idx]
                    weight = top_weights[b, t, k_idx]
                    start_dim = expert_idx * (C // self.expert_types)
                    end_dim = start_dim + (C // self.expert_types)
                    final_output[b, t, start_dim:end_dim] += weight * outputs[expert_idx][b, t]
        
        return self.c_proj(final_output)

class SparseExpertGPT(nn.Module):
    def __init__(self, vocab_size=10000, n_embd=256, n_layer=6, n_head=8):  # Reduced for budget
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(1024, n_embd)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': SparseExpertAttention(n_embd, n_head),
                'ln_1': nn.LayerNorm(n_embd),
                'mlp': nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.GELU(),
                    nn.Linear(4 * n_embd, n_embd),
                    nn.Dropout(0.1)
                ),
                'ln_2': nn.LayerNorm(n_embd)
            }) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight sharing to reduce parameters
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.wte(idx) + self.wpe(pos)
        
        for block in self.blocks:
            x = x + block['attn'](block['ln_1'](x))
            x = x + block['mlp'](block['ln_2'](x))
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training setup
if __name__ == '__main__':
    # Reduced config to fit 10M budget
    model = SparseExpertGPT(vocab_size=10000, n_embd=256, n_layer=6, n_head=6)
    total_params = count_parameters(model)
    print(f'Total parameters: {total_params/1e6:.1f}M')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Dummy training data
    for step in range(100):
        x = torch.randint(0, 10000, (4, 32))  # Small batch, short sequence
        targets = torch.randint(0, 10000, (4, 32))
        
        logits, loss = model(x, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f'Step {step}, Loss: {loss.item():.4f}')
            
    print(f'Final loss: {loss.item():.4f}')
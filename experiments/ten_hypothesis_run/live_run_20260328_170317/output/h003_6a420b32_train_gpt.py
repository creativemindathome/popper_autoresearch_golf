import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn import functional as F

class GradientGuidedAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, sparsity_ratio=0.3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads
        self.sparsity_ratio = sparsity_ratio
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Gradient importance tracking
        self.register_buffer('grad_importance', torch.ones(seq_len))
        self.importance_decay = 0.95
        self.grad_scale = 100.0
        
    def update_importance_scores(self, x_grad):
        if x_grad is not None:
            # Compute gradient magnitude per position
            pos_grad_mag = torch.norm(x_grad, dim=-1).mean(0)  # [seq_len]
            # Update exponential moving average
            self.grad_importance = self.importance_decay * self.grad_importance + \
                                 (1 - self.importance_decay) * pos_grad_mag.detach()
    
    def create_sparse_mask(self, seq_len, batch_size):
        # Create attention mask based on gradient importance
        importance_probs = F.softmax(self.grad_importance[:seq_len] * self.grad_scale, dim=0)
        
        # Each position gets attention based on importance
        keep_prob = 1 - self.sparsity_ratio
        base_connections = int(seq_len * keep_prob)
        
        mask = torch.zeros(seq_len, seq_len, device=self.grad_importance.device)
        
        for i in range(seq_len):
            # Always attend to self and local neighbors
            mask[i, max(0, i-1):min(seq_len, i+2)] = 1
            
            # Add importance-weighted random connections
            n_extra = max(0, base_connections - 3)  # minus local connections
            if n_extra > 0:
                # Sample based on importance scores
                candidates = torch.arange(seq_len, device=mask.device)
                weights = importance_probs.clone()
                weights[max(0, i-1):min(seq_len, i+2)] = 0  # exclude already connected
                
                if weights.sum() > 0:
                    selected = torch.multinomial(weights, min(n_extra, (weights > 0).sum()), replacement=False)
                    mask[i, selected] = 1
        
        return mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply gradient-guided sparse mask
        sparse_mask = self.create_sparse_mask(T, B)
        scores = scores.masked_fill(sparse_mask.unsqueeze(1) == 0, float('-inf'))
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class GradientGuidedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.attn = GradientGuidedAttention(d_model, n_heads, seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x_norm = self.ln1(x)
        x = x + self.attn(x_norm)
        x = x + self.mlp(self.ln2(x))
        return x

class GradientGuidedGPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=384, n_heads=6, n_layers=6, seq_len=256):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([GradientGuidedTransformerBlock(d_model, n_heads, seq_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        
        x = self.tok_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        return self.head(x)
    
    def update_gradient_importance(self):
        # Update importance scores in all attention layers
        for block in self.blocks:
            if hasattr(block.attn.tok_emb, 'grad') and block.attn.tok_emb.grad is not None:
                block.attn.update_importance_scores(block.attn.tok_emb.grad)

# Training code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GradientGuidedGPT().to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Dummy data for micro-training
seq_len = 256
batch_size = 4
num_steps = 1000

losses = []
for step in range(num_steps):
    # Generate random data
    x = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    y = torch.cat([x[:, 1:], torch.randint(0, 50257, (batch_size, 1), device=device)], dim=1)
    
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    
    # Update gradient importance scores
    model.update_gradient_importance()
    
    optimizer.step()
    losses.append(loss.item())
    
    if step % 100 == 0:
        print(f'Step {step}, Loss: {loss.item():.4f}')

print(f'Final loss: {losses[-1]:.4f}, Improvement: {losses[0] - losses[-1]:.4f}')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math

class GradientSculptedAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, sparsity_target=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.sparsity_target = sparsity_target
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        # Gradient tracking
        self.register_buffer('grad_ema', torch.zeros(1))
        self.register_buffer('step_count', torch.zeros(1))
        self.grad_decay = 0.99
        
    def forward(self, x):
        b, n, d = x.shape
        h = self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, self.dim_head).transpose(1, 2), qkv)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if self.training:
            # Apply gradient-based masking
            current_sparsity = self.sparsity_target * min(1.0, self.step_count.item() / 1000)
            if current_sparsity > 0:
                # Create mask based on gradient EMA
                threshold = torch.quantile(self.grad_ema.flatten(), 1 - current_sparsity)
                mask = self.grad_ema >= threshold
                dots = dots.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(dots, dim=-1)
        
        # Track gradients for attention weights
        if self.training and attn.requires_grad:
            def grad_hook(grad):
                grad_sq = grad.pow(2)
                if self.grad_ema.numel() != grad_sq.numel():
                    self.grad_ema.data = torch.zeros_like(grad_sq)
                self.grad_ema.data = self.grad_decay * self.grad_ema + (1 - self.grad_decay) * grad_sq
                self.step_count.data += 1
            
            attn.register_hook(grad_hook)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class GSATransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = GradientSculptedAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GSATransformer(nn.Module):
    def __init__(self, vocab_size=50257, dim=512, depth=6, heads=8, max_seq_len=1024):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        self.blocks = nn.ModuleList([
            GSATransformerBlock(dim, heads) for _ in range(depth)
        ])
        
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(0, t, device=x.device).unsqueeze(0)
        
        x = self.token_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        return self.head(x)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GSATransformer(vocab_size=1000, dim=256, depth=4, heads=4, max_seq_len=128).to(device)
optimizer = Adam(model.parameters(), lr=3e-4)

# Synthetic data for micro-training
def generate_batch(batch_size=32, seq_len=64, vocab_size=1000):
    return torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

# Training loop
model.train()
losses = []

for step in range(500):
    batch = generate_batch()
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    losses.append(loss.item())
    
    if step % 50 == 0:
        avg_sparsity = sum(block.attn.sparsity_target * min(1.0, block.attn.step_count.item() / 1000) 
                          for block in model.blocks) / len(model.blocks)
        print(f'Step {step}, Loss: {loss.item():.4f}, Avg Sparsity: {avg_sparsity:.3f}')

print(f'Final loss: {losses[-1]:.4f}')
print(f'Loss reduction: {losses[0] - losses[-1]:.4f}')
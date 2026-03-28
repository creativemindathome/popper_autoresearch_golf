import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import math

class AdaptiveExitAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class ExitLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.classifier = nn.Linear(d_model, vocab_size)
        self.confidence = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        logits = self.classifier(x)
        conf = self.confidence(x)
        return logits, conf

class AdaptiveDepthLayer(nn.Module):
    def __init__(self, d_model, n_heads, vocab_size):
        super().__init__()
        self.attn = AdaptiveExitAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.exit_head = ExitLayer(d_model, vocab_size)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        logits, confidence = self.exit_head(x)
        return x, logits, confidence

class AdaptiveDepthTransformer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(1024, d_model)
        self.layers = nn.ModuleList([
            AdaptiveDepthLayer(d_model, n_heads, vocab_size) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Learnable exit thresholds
        self.exit_thresholds = nn.Parameter(torch.ones(n_layers) * 0.5)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.wte(idx) + self.wpe(pos)
        
        exit_losses = []
        exit_counts = torch.zeros(self.n_layers, device=idx.device)
        
        for i, layer in enumerate(self.layers):
            x, layer_logits, confidence = layer(x)
            
            if targets is not None:
                # Calculate loss at this exit point
                exit_loss = F.cross_entropy(layer_logits.view(-1, self.vocab_size), 
                                          targets.view(-1), ignore_index=-1)
                exit_losses.append(exit_loss)
                
                # Count tokens that would exit here
                would_exit = (confidence.squeeze(-1) > self.exit_thresholds[i]).float()
                exit_counts[i] = would_exit.sum()
        
        # Final layer processing
        x = self.ln_f(x)
        final_logits = self.lm_head(x)
        
        if targets is not None:
            # Main loss
            main_loss = F.cross_entropy(final_logits.view(-1, self.vocab_size), 
                                      targets.view(-1), ignore_index=-1)
            
            # Combine losses with decreasing weights for earlier exits
            total_loss = main_loss
            for i, exit_loss in enumerate(exit_losses):
                weight = 0.5 ** (self.n_layers - i)  # Earlier exits get lower weight
                total_loss += weight * exit_loss
            
            # Efficiency penalty - encourage early exits
            avg_depth = sum(i * count for i, count in enumerate(exit_counts)) / (sum(exit_counts) + 1e-8)
            efficiency_penalty = 0.1 * avg_depth / self.n_layers
            total_loss += efficiency_penalty
            
            return final_logits, total_loss
        
        return final_logits

# Training setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AdaptiveDepthTransformer().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Dummy data for testing
vocab_size = 50257
seq_len = 128
batch_size = 4

for step in range(100):
    # Generate random data
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    
    if step % 20 == 0:
        print(f'Step {step}, Loss: {loss.item():.4f}')
        # Print exit threshold evolution
        thresholds = model.exit_thresholds.data
        print(f'Exit thresholds: {thresholds.tolist()}')

print('Training completed successfully!')
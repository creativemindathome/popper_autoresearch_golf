import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalDepthGPT(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=768, n_head=12, n_layer=12, max_seq_len=1024):
        super().__init__()
        self.n_layer = n_layer
        self.active_layers = 2  # Start with only 2 layers active
        self.gradient_threshold = 1e-3
        self.patience_counter = 0
        self.patience_limit = 100  # Steps to wait before adding layer
        
        # Token and position embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_seq_len, n_embd)
        
        # All transformer blocks (but not all will be active initially)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])
        
        # Layer norm and output head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Track gradient magnitudes for depth decisions
        self.grad_magnitudes = []
        
    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        # Embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        # Forward through only active layers
        for i in range(min(self.active_layers, self.n_layer)):
            x = self.blocks[i](x)
        
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits
    
    def update_depth(self):
        """Called after backward pass to potentially increase depth"""
        if self.active_layers >= self.n_layer:
            return
        
        # Calculate average gradient magnitude in active layers
        total_grad_mag = 0
        param_count = 0
        for i in range(self.active_layers):
            for param in self.blocks[i].parameters():
                if param.grad is not None:
                    total_grad_mag += param.grad.abs().mean().item()
                    param_count += 1
        
        if param_count > 0:
            avg_grad_mag = total_grad_mag / param_count
            self.grad_magnitudes.append(avg_grad_mag)
            
            # If gradients are small, we might have saturated current depth
            if avg_grad_mag < self.gradient_threshold:
                self.patience_counter += 1
                if self.patience_counter >= self.patience_limit:
                    self.active_layers = min(self.active_layers + 1, self.n_layer)
                    self.patience_counter = 0
                    print(f"Increased active layers to {self.active_layers}")
            else:
                self.patience_counter = 0

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        
    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

# Training loop
def train_temporal_depth_gpt():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalDepthGPT(vocab_size=1000, n_embd=256, n_head=8, n_layer=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    # Dummy data
    batch_size, seq_len = 4, 128
    
    losses = []
    active_layers_history = []
    
    for step in range(1000):
        # Generate random data
        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        y = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        
        # Update depth based on gradient analysis
        model.update_depth()
        
        optimizer.step()
        
        losses.append(loss.item())
        active_layers_history.append(model.active_layers)
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, Active layers: {model.active_layers}")
    
    return losses, active_layers_history

if __name__ == '__main__':
    losses, active_layers_history = train_temporal_depth_gpt()
    print(f"Final active layers: {active_layers_history[-1]}")
    print(f"Final loss: {losses[-1]:.4f}")
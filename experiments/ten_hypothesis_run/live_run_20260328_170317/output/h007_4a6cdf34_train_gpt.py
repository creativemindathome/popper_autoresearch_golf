import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np

class GradientInformedAttention(nn.Module):
    def __init__(self, d_model, n_heads, gradient_memory_size=100, gradient_weight=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.gradient_memory_size = gradient_memory_size
        self.gradient_weight = gradient_weight
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Gradient memory buffers (not parameters)
        self.register_buffer('gradient_memory', torch.zeros(gradient_memory_size, 512))  # max seq len 512
        self.register_buffer('memory_index', torch.zeros(1, dtype=torch.long))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Standard QKV computation
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Gradient-based bias computation
        if self.training and x.requires_grad:
            # Compute gradient norms for current tokens
            grad_norms = self._compute_gradient_bias(x, seq_len)
            # Add gradient bias to attention scores
            scores = scores + self.gradient_weight * grad_norms.unsqueeze(1).unsqueeze(1)
        
        # Apply softmax and attention
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_linear(attended)
        
        # Store gradients for next iteration (during backward pass)
        if self.training:
            output.register_hook(self._gradient_hook)
        
        return output
    
    def _compute_gradient_bias(self, x, seq_len):
        # Use historical gradient information
        recent_grads = self.gradient_memory[:min(self.memory_index.item(), self.gradient_memory_size)]
        if recent_grads.shape[0] > 0:
            avg_grad_norms = recent_grads.mean(dim=0)[:seq_len]
            return avg_grad_norms.unsqueeze(0)  # broadcast for batch
        else:
            return torch.zeros(1, seq_len, device=x.device)
    
    def _gradient_hook(self, grad):
        if grad is not None:
            # Compute L2 norm of gradients for each position
            grad_norms = torch.norm(grad, dim=-1).mean(dim=0)  # average over batch
            
            # Update gradient memory in circular buffer
            idx = self.memory_index.item() % self.gradient_memory_size
            if grad_norms.shape[0] <= self.gradient_memory.shape[1]:
                self.gradient_memory[idx, :grad_norms.shape[0]] = grad_norms.detach()
                self.memory_index.add_(1)

class GIATransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = GradientInformedAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class GIATransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
        
        self.layers = nn.ModuleList([
            GIATransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.shape[1]
        
        # Embedding and positional encoding
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        return self.head(x)

# Simple dataset for micro-training
class SimpleTextDataset(Dataset):
    def __init__(self, vocab_size=1000, seq_len=64, size=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data = torch.randint(0, vocab_size, (size, seq_len))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 1000
model = GIATransformer(vocab_size=vocab_size, d_model=256, n_heads=8, n_layers=4).to(device)

# Dataset and dataloader
dataset = SimpleTextDataset(vocab_size=vocab_size, seq_len=64, size=2000)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
losses = []

for epoch in range(10):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        # Language modeling: predict next token
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')

print('Training completed!')
print(f'Final loss trajectory: {losses}')
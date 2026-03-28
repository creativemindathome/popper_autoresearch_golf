import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class FractalAttention(nn.Module):
    def __init__(self, d_model, n_heads, chunk_sizes=[1, 4, 16]):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_sizes = chunk_sizes
        self.head_dim = d_model // n_heads
        
        # Separate attention layers for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads // len(chunk_sizes), batch_first=True)
            for _ in chunk_sizes
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(len(chunk_sizes)))
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        outputs = []
        
        for i, (chunk_size, attn_layer) in enumerate(zip(self.chunk_sizes, self.scale_attentions)):
            if chunk_size == 1:
                # Fine-grained: standard token-level attention
                out, _ = attn_layer(x, x, x, attn_mask=mask)
                outputs.append(out)
            else:
                # Coarse-grained: chunk-based attention
                # Pad sequence to be divisible by chunk_size
                pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
                if pad_len > 0:
                    x_padded = F.pad(x, (0, 0, 0, pad_len))
                else:
                    x_padded = x
                    
                # Reshape to chunks and apply attention
                chunks = x_padded.view(batch_size, -1, chunk_size * d_model)
                chunk_repr = chunks.mean(dim=2, keepdim=True).expand(-1, -1, chunk_size * d_model)
                chunk_repr = chunk_repr.view(batch_size, -1, d_model)
                
                out, _ = attn_layer(chunk_repr, chunk_repr, chunk_repr)
                
                # Expand back to original sequence length
                out = out.repeat_interleave(chunk_size, dim=1)[:, :seq_len, :]
                outputs.append(out)
        
        # Weighted combination of scales
        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * out for w, out in zip(weights, outputs))
        return self.output_proj(combined)

class FractalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, chunk_sizes=[1, 4, 16]):
        super().__init__()
        self.attention = FractalAttention(d_model, n_heads, chunk_sizes)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x

class FractalGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_heads=8, n_layers=4, max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Adaptive chunk sizes based on layer depth
        self.blocks = nn.ModuleList([
            FractalTransformerBlock(d_model, n_heads, d_model * 4, 
                                  chunk_sizes=[1, 2**(i+1), 2**(i+2)])
            for i in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, targets=None):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device)
        
        x = self.embedding(x) + self.pos_embedding[:seq_len]
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits

# Training setup
def generate_synthetic_data(vocab_size=1000, seq_len=64, num_samples=1000):
    # Create synthetic sequential data with some structure
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return data

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    vocab_size = 1000
    seq_len = 64
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 100
    
    # Generate data
    data = generate_synthetic_data(vocab_size, seq_len, 1000)
    targets = torch.roll(data, -1, dims=1)  # Next token prediction
    
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = FractalGPT(vocab_size=vocab_size, d_model=256, n_heads=8, n_layers=4, max_seq_len=seq_len)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (batch_data, batch_targets) in enumerate(dataloader):
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(batch_data, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    print("Training completed")
    return model

if __name__ == "__main__":
    model = train_model()
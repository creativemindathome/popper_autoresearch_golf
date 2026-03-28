import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class GradientConditionedAttention(nn.Module):
    def __init__(self, d_model, n_heads, gradient_decay=0.99):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.gradient_decay = gradient_decay
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Gradient tracking buffers
        self.register_buffer('k_grad_ema', torch.zeros(1024, d_model))  # max seq len
        self.register_buffer('v_grad_ema', torch.zeros(1024, d_model))
        self.register_buffer('step_count', torch.tensor(0))
        
        # Importance projection
        self.importance_proj = nn.Linear(d_model, n_heads)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Store k,v for gradient tracking
        self.current_k = k.clone().detach().requires_grad_(True)
        self.current_v = v.clone().detach().requires_grad_(True)
        self.seq_len = seq_len
        
        # Compute standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores.masked_fill_(mask, float('-inf'))
        
        # Compute gradient-based importance scores
        k_importance = self.importance_proj(self.k_grad_ema[:seq_len]).transpose(0, 1)  # n_heads x seq_len
        v_importance = self.importance_proj(self.v_grad_ema[:seq_len]).transpose(0, 1)
        
        # Combine importance scores and add to attention
        importance_bias = (k_importance.unsqueeze(-1) + v_importance.unsqueeze(-2)) / 2
        scores = scores + 0.1 * importance_bias.unsqueeze(0)  # scale factor
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)
    
    def update_gradient_stats(self):
        if hasattr(self, 'current_k') and self.current_k.grad is not None:
            # Update EMA of squared gradients
            k_grad_norm = torch.norm(self.current_k.grad, dim=(0, 2, 3))  # seq_len
            v_grad_norm = torch.norm(self.current_v.grad, dim=(0, 2, 3))
            
            seq_len = min(self.seq_len, self.k_grad_ema.size(0))
            self.k_grad_ema[:seq_len] = (self.gradient_decay * self.k_grad_ema[:seq_len] + 
                                       (1 - self.gradient_decay) * k_grad_norm[:seq_len].unsqueeze(1))
            self.v_grad_ema[:seq_len] = (self.gradient_decay * self.v_grad_ema[:seq_len] + 
                                       (1 - self.gradient_decay) * v_grad_norm[:seq_len].unsqueeze(1))

class GCATransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = GradientConditionedAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class GCATransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_heads=8, n_layers=6, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([GCATransformerBlock(d_model, n_heads, d_model*4) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:seq_len]
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return self.lm_head(x)
    
    def update_all_gradient_stats(self):
        for block in self.blocks:
            block.attn.update_gradient_stats()

# Training setup
def create_dummy_data(vocab_size=1000, seq_len=128, num_samples=1000):
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return data

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model setup
    model = GCATransformer(vocab_size=1000, d_model=256, n_heads=8, n_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Data
    train_data = create_dummy_data()
    dataloader = DataLoader(TensorDataset(train_data), batch_size=16, shuffle=True)
    
    losses = []
    
    model.train()
    for epoch in range(20):
        epoch_loss = 0
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Update gradient statistics BEFORE optimizer step
            model.update_all_gradient_stats()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}')
    
    return losses

if __name__ == '__main__':
    losses = train_model()
    print('Training completed!')
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset

class DifferentialAttentionHead(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_k)
        
    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        
        # Compute attention entropy
        eps = 1e-8
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1).mean()
        
        output = attn_weights @ v
        return output, attn_weights, entropy

class DifferentialAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, head_pool_size=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_pool_size = head_pool_size
        self.d_k = d_model // n_heads
        
        # Pool of attention heads (more than we'll use)
        self.head_pool = nn.ModuleList([
            DifferentialAttentionHead(d_model, self.d_k) 
            for _ in range(head_pool_size)
        ])
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(d_model, head_pool_size),
            nn.LayerNorm(head_pool_size),
            nn.ReLU(),
            nn.Linear(head_pool_size, head_pool_size)
        )
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Compute routing scores based on input
        route_scores = self.router(x.mean(dim=1))  # [batch, head_pool_size]
        
        # Select top-k heads using Gumbel softmax for differentiability
        temperature = 0.5
        gumbel_noise = torch.rand_like(route_scores).log().neg().log().neg()
        route_logits = (route_scores + gumbel_noise) / temperature
        route_weights = F.softmax(route_logits, dim=-1)
        
        # Get top-k head indices
        _, top_indices = torch.topk(route_weights, self.n_heads, dim=-1)
        
        outputs = []
        entropies = []
        
        for i in range(batch_size):
            batch_outputs = []
            batch_entropies = []
            
            for head_idx in top_indices[i]:
                head_out, _, entropy = self.head_pool[head_idx](x[i:i+1])
                batch_outputs.append(head_out)
                batch_entropies.append(entropy)
            
            outputs.append(torch.cat(batch_outputs, dim=-1))
            entropies.extend(batch_entropies)
        
        # Combine outputs
        multi_head_out = torch.cat(outputs, dim=0)
        output = self.output_proj(multi_head_out)
        
        # Residual connection and normalization
        output = self.norm(x + output)
        
        avg_entropy = sum(entropies) / len(entropies)
        return output, avg_entropy

class DifferentialTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = DifferentialAttentionLayer(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_out, entropy = self.attention(x)
        ff_out = self.feed_forward(attn_out)
        output = self.norm(attn_out + ff_out)
        return output, entropy

class DifferentialGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=6, n_heads=8, d_ff=1024, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        self.blocks = nn.ModuleList([
            DifferentialTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.shape[1]
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding[:seq_len]
        x = token_emb + pos_emb
        
        entropies = []
        for block in self.blocks:
            x, entropy = block(x)
            entropies.append(entropy)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, entropies

# Training script
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create synthetic data
vocab_size = 1000
seq_len = 64
batch_size = 16
n_samples = 1000

data = torch.randint(0, vocab_size, (n_samples, seq_len))
targets = torch.roll(data, -1, dims=1)
dataset = TensorDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = DifferentialGPT(vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=512, max_seq_len=seq_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training loop
model.train()
for epoch in range(100):
    total_loss = 0
    total_entropy_loss = 0
    
    for batch_idx, (data_batch, targets_batch) in enumerate(dataloader):
        data_batch, targets_batch = data_batch.to(device), targets_batch.to(device)
        
        optimizer.zero_grad()
        
        logits, entropies = model(data_batch)
        
        # Language modeling loss
        lm_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets_batch.reshape(-1))
        
        # Entropy regularization (encourage diverse, low-entropy attention)
        entropy_loss = sum(entropies) / len(entropies)
        entropy_regularization = 0.01 * entropy_loss  # Penalize high entropy
        
        total_loss_val = lm_loss + entropy_regularization
        total_loss_val.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += lm_loss.item()
        total_entropy_loss += entropy_loss.item()
    
    scheduler.step()
    
    if epoch % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        avg_entropy = total_entropy_loss / len(dataloader)
        print(f'Epoch {epoch}: Loss = {avg_loss:.4f}, Avg Entropy = {avg_entropy:.4f}')

print('Training completed!')
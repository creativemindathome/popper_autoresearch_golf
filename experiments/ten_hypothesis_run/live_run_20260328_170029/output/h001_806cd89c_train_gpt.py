import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EntropyPredictor(nn.Module):
    def __init__(self, d_model, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Conv1d(d_model, 64, kernel_size=window_size, padding=window_size//2)
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x_conv = self.conv(x.transpose(1, 2))  # (batch, 64, seq_len)
        x_pooled = F.adaptive_avg_pool1d(x_conv, x_conv.size(2))
        entropy_scores = self.predictor(x_pooled.transpose(1, 2))
        return entropy_scores.squeeze(-1)  # (batch, seq_len)

class EntropyGuidedAttention(nn.Module):
    def __init__(self, d_model, n_heads, entropy_threshold=0.3, sparse_ratio=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.entropy_threshold = entropy_threshold
        self.sparse_ratio = sparse_ratio
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.entropy_pred = EntropyPredictor(d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Predict entropy for each position
        entropy_scores = self.entropy_pred(x)  # (B, T)
        
        # Decide computation path per token
        high_entropy_mask = entropy_scores > self.entropy_threshold
        
        # Fast path: direct residual for low-entropy tokens
        output = x.clone()
        
        # Full attention path for high-entropy tokens
        if high_entropy_mask.any():
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention only for high-entropy positions
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Sparse attention: mask out low-scoring pairs
            if self.training:
                topk = max(1, int(T * self.sparse_ratio))
                top_scores, _ = torch.topk(scores, topk, dim=-1)
                threshold = top_scores[..., -1:]
                sparse_mask = scores >= threshold
                scores = scores.masked_fill(~sparse_mask, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
            attn_output = self.out_proj(attn_output)
            
            # Apply attention output only to high-entropy positions
            high_entropy_mask_expanded = high_entropy_mask.unsqueeze(-1).expand_as(output)
            output = torch.where(high_entropy_mask_expanded, attn_output, output)
        
        return output, entropy_scores

class EGATransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = EntropyGuidedAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_out, entropy_scores = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, entropy_scores

class EGATransformer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([EGATransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        
        x = self.token_emb(x) + self.pos_emb(pos)
        
        all_entropy_scores = []
        for block in self.blocks:
            x, entropy_scores = block(x)
            all_entropy_scores.append(entropy_scores)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits, torch.stack(all_entropy_scores)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EGATransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Simple training loop with entropy regularization
for step in range(1000):
    # Generate dummy data
    batch_size, seq_len = 4, 128
    x = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
    targets = torch.roll(x, -1, dims=1)
    
    logits, entropy_scores = model(x)
    
    # Main language modeling loss
    lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    # Entropy diversity regularization (encourage varied entropy predictions)
    entropy_diversity = -torch.var(entropy_scores.mean(dim=0))  # Want high variance across positions
    
    total_loss = lm_loss + 0.01 * entropy_diversity
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f'Step {step}, Loss: {lm_loss.item():.4f}, Entropy Div: {entropy_diversity.item():.4f}')
        print(f'Avg Entropy Score: {entropy_scores.mean().item():.3f}')
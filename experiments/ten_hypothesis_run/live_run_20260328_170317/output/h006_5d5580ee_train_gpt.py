import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

class UncertaintyEstimator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ema_decay = 0.99
        self.register_buffer('grad_ema', torch.zeros(1))
        self.uncertainty_head = nn.Linear(d_model, 1)
        
    def forward(self, x, training=True):
        if training and x.requires_grad:
            # Estimate uncertainty from gradient magnitude
            uncertainty_raw = self.uncertainty_head(x)
            uncertainty = torch.sigmoid(uncertainty_raw).squeeze(-1)
            
            # Update EMA of gradient magnitude
            if self.grad_ema.sum() > 0:
                grad_mag = torch.norm(torch.autograd.grad(uncertainty.sum(), x, retain_graph=True)[0], dim=-1)
                self.grad_ema = self.ema_decay * self.grad_ema + (1 - self.ema_decay) * grad_mag.mean()
            
            return uncertainty
        else:
            return torch.sigmoid(self.uncertainty_head(x)).squeeze(-1)

class AdaptiveMoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Uncertainty-based routing
        self.uncertainty_estimator = UncertaintyEstimator(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Estimate uncertainty for each token
        uncertainty = self.uncertainty_estimator(x, self.training)
        
        # Determine number of experts per token based on uncertainty
        # High uncertainty -> more experts (up to 4)
        num_experts_per_token = torch.clamp(torch.ceil(uncertainty * self.num_experts), 1, self.num_experts).long()
        
        output = torch.zeros_like(x)
        
        for i in range(1, self.num_experts + 1):
            # Mask for tokens that should use i experts
            mask = (num_experts_per_token >= i)
            if not mask.any():
                continue
                
            # Select expert based on token position and uncertainty
            expert_idx = ((torch.arange(seq_len, device=x.device).unsqueeze(0) + i - 1) % self.num_experts)
            
            for j in range(self.num_experts):
                expert_mask = mask & (expert_idx == j)
                if expert_mask.any():
                    expert_input = x[expert_mask]
                    expert_output = self.experts[j](expert_input)
                    
                    # Weight by normalized uncertainty
                    weight = uncertainty[expert_mask].unsqueeze(-1) / i
                    output[expert_mask] += expert_output * weight
        
        return output

class AdaptiveMoEBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, num_experts=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = AdaptiveMoELayer(d_model, d_ff, num_experts)
        
    def forward(self, x, attn_mask=None):
        # Attention
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)
        x = x + attn_out
        
        # Adaptive MoE
        moe_out = self.moe(self.ln2(x))
        x = x + moe_out
        
        return x

class AdaptiveMoETransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_heads=6, n_layers=6, d_ff=1536, max_len=512, num_experts=4):
        super().__init__()
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            AdaptiveMoEBlock(d_model, d_ff, n_heads, num_experts) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x, targets=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        
        # Embeddings
        x = self.token_emb(x) + self.pos_emb(pos)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss

# Training code
def train_adaptive_moe():
    # Simple text data (character-level)
    text = "hello world this is a test of the adaptive mixture of experts architecture that dynamically allocates computation based on uncertainty estimates" * 100
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    # Prepare data
    seq_len = 128
    data = [char_to_idx[ch] for ch in text]
    sequences = []
    for i in range(0, len(data) - seq_len, seq_len//4):
        sequences.append(data[i:i+seq_len])
    
    sequences = torch.tensor(sequences[:100])  # Limit for micro-training
    
    # Create dataset
    dataset = TensorDataset(sequences[:, :-1], sequences[:, 1:])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Model and optimizer
    model = AdaptiveMoETransformer(vocab_size=len(chars), d_model=256, n_heads=4, n_layers=4, num_experts=4)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    model.train()
    losses = []
    
    print(f"Training Adaptive MoE Transformer with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    for epoch in range(50):
        epoch_losses = []
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            logits, loss = model(input_ids, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return losses

if __name__ == "__main__":
    losses = train_adaptive_moe()
    print(f"Final loss: {losses[-1]:.4f}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class DynamicDepthAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn.masked_fill_(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class RoutingGate(nn.Module):
    def __init__(self, d_model, threshold=0.5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        self.threshold = threshold
        
    def forward(self, x):
        # x: (B, T, d_model)
        confidence = self.gate(x)  # (B, T, 1)
        should_continue = (confidence < self.threshold).float()
        return confidence, should_continue

class DynamicDepthBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, is_exit_layer=False):
        super().__init__()
        self.attn = DynamicDepthAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.router = RoutingGate(d_model) if not is_exit_layer else None
        self.is_exit_layer = is_exit_layer
        
    def forward(self, x, mask=None, routing_info=None):
        # x: (B, T, d_model)
        # routing_info: dict with 'active_mask' indicating which tokens are still processing
        
        if routing_info is not None:
            active_mask = routing_info['active_mask']  # (B, T, 1)
        else:
            active_mask = torch.ones_like(x[:, :, :1])
            
        # Only process active tokens
        residual = x
        x = self.ln1(x)
        x = self.attn(x, mask)
        x = residual + x * active_mask  # Only update active tokens
        
        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = residual + x * active_mask
        
        # Routing decision
        if self.router is not None and not self.is_exit_layer:
            confidence, should_continue = self.router(x)
            new_active_mask = active_mask * should_continue
            routing_info = {
                'active_mask': new_active_mask,
                'confidence': confidence,
                'exit_tokens': active_mask - new_active_mask
            }
        else:
            routing_info = {'active_mask': active_mask}
            
        return x, routing_info

class DynamicDepthTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=12, d_ff=2048, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            DynamicDepthBlock(d_model, n_heads, d_ff, dropout, is_exit_layer=(i == n_layers-1))
            for i in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Sparsity penalty weight
        self.sparsity_weight = 0.01
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x, targets=None):
        B, T = x.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        # Token + position embeddings
        pos = torch.arange(T, device=x.device)
        x = self.token_emb(x) + self.pos_emb(pos)
        
        # Track routing statistics
        total_confidence = 0
        exit_stats = []
        
        routing_info = None
        
        # Pass through dynamic depth blocks
        for i, block in enumerate(self.blocks):
            x, routing_info = block(x, mask, routing_info)
            
            if routing_info.get('confidence') is not None:
                total_confidence += routing_info['confidence'].mean()
                if 'exit_tokens' in routing_info:
                    exit_stats.append(routing_info['exit_tokens'].sum())
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Add sparsity penalty to encourage early exits
            if total_confidence > 0:
                sparsity_penalty = self.sparsity_weight * total_confidence
                loss += sparsity_penalty
                
        return {
            'logits': logits,
            'loss': loss,
            'routing_stats': {
                'avg_confidence': total_confidence / len([b for b in self.blocks if b.router is not None]),
                'exit_stats': exit_stats
            }
        }

def train_dynamic_depth_transformer():
    # Hyperparameters
    vocab_size = 10000
    d_model = 256
    n_heads = 8
    n_layers = 6
    d_ff = 1024
    batch_size = 32
    seq_len = 128
    lr = 3e-4
    n_steps = 1000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DynamicDepthTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=seq_len
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    model.train()
    for step in range(n_steps):
        # Generate random data
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.roll(x, -1, dims=1)
        
        optimizer.zero_grad()
        
        start_time = time.time()
        output = model(x, targets)
        loss = output['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        forward_time = time.time() - start_time
        
        if step % 100 == 0:
            routing_stats = output['routing_stats']
            print(f"Step {step}, Loss: {loss.item():.4f}, "
                  f"Avg Confidence: {routing_stats['avg_confidence']:.4f}, "
                  f"Time: {forward_time:.3f}s")
            
            # Print exit statistics
            if routing_stats['exit_stats']:
                exits_per_layer = [stat.item() for stat in routing_stats['exit_stats']]
                print(f"  Exits per layer: {exits_per_layer}")
    
    return model

if __name__ == '__main__':
    model = train_dynamic_depth_transformer()
    print("Training completed!")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EntropyGate(nn.Module):
    def __init__(self, d_model, init_threshold=2.0):
        super().__init__()
        self.threshold_param = nn.Parameter(torch.tensor(init_threshold))
        self.gate_proj = nn.Linear(d_model, 1)
        self.entropy_ema = 0.95  # EMA decay for entropy tracking
        
    def forward(self, x, entropy_history):
        # Compute gating probability based on entropy
        gate_logits = self.gate_proj(x).squeeze(-1)  # [batch, seq]
        threshold = torch.sigmoid(self.threshold_param) * 4.0  # Threshold in [0, 4]
        
        # Gate based on entropy history - high entropy = more computation
        gate_prob = torch.sigmoid((entropy_history - threshold) * 2.0)
        
        # Stochastic gating during training, deterministic during eval
        if self.training:
            gate = torch.bernoulli(gate_prob)
        else:
            gate = (gate_prob > 0.5).float()
            
        return gate, gate_prob.mean()  # Return average gate rate for monitoring

class EntropyGatedAttention(nn.Module):
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
        
    def forward(self, x, mask=None, gate=None):
        batch_size, seq_len, d_model = x.shape
        
        # If gate is provided, only compute for active tokens
        if gate is not None:
            active_mask = gate.unsqueeze(-1)  # [batch, seq, 1]
            # For inactive tokens, return identity (residual connection only)
            output = x.clone()
            
            # Only compute attention for active tokens
            if active_mask.sum() > 0:
                q = self.q_proj(x) * active_mask
                k = self.k_proj(x) * active_mask  
                v = self.v_proj(x) * active_mask
                
                q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float('-inf'))
                    
                attn = F.softmax(scores, dim=-1)
                attn = self.dropout(attn)
                
                out = torch.matmul(attn, v)
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
                out = self.out_proj(out)
                
                # Apply attention output only to active tokens
                output = output * (1 - active_mask) + out * active_mask
        else:
            # Standard attention computation
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
                
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            output = self.out_proj(out)
            
        return output

class EntropyGatedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = EntropyGatedAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Entropy gate - becomes more aggressive in later layers
        self.entropy_gate = EntropyGate(d_model, init_threshold=1.0 + layer_idx * 0.3)
        
    def forward(self, x, entropy_history, mask=None):
        # Compute gating decision based on entropy
        gate, gate_rate = self.entropy_gate(x, entropy_history)
        
        # Self-attention with gating
        attn_out = self.attention(self.norm1(x), mask=mask, gate=gate)
        x = x + attn_out
        
        # Feed-forward with gating
        if gate is not None:
            active_mask = gate.unsqueeze(-1)
            ff_out = self.ff(self.norm2(x)) * active_mask
        else:
            ff_out = self.ff(self.norm2(x))
            
        x = x + ff_out
        
        return x, gate_rate

class EntropyGatedGPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12, 
                 d_ff=3072, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            EntropyGatedTransformerBlock(d_model, n_heads, d_ff, dropout, i)
            for i in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Entropy tracking
        self.register_buffer('entropy_ema_coef', torch.tensor(0.9))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token and position embeddings
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        
        # Initial entropy (high uncertainty)
        entropy_history = torch.full((batch_size, seq_len), 3.0, device=device)
        
        gate_rates = []
        
        # Forward through layers with entropy gating
        for layer in self.layers:
            x, gate_rate = layer(x, entropy_history, mask=mask)
            gate_rates.append(gate_rate)
            
            # Update entropy history based on current predictions
            if self.training:
                with torch.no_grad():
                    logits = self.lm_head(self.ln_f(x))
                    probs = F.softmax(logits, dim=-1)
                    current_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                    
                    # EMA update of entropy history
                    entropy_history = (self.entropy_ema_coef * entropy_history + 
                                     (1 - self.entropy_ema_coef) * current_entropy)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return {
            'logits': logits,
            'loss': loss,
            'gate_rates': gate_rates,
            'avg_gate_rate': sum(gate_rates) / len(gate_rates)
        }

# Training code
if __name__ == '__main__':
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Hyperparameters
    vocab_size = 1000
    d_model = 256
    n_heads = 8
    n_layers = 6
    d_ff = 1024
    seq_len = 128
    batch_size = 8
    lr = 3e-4
    num_epochs = 5
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EntropyGatedGPT(vocab_size, d_model, n_heads, n_layers, d_ff, seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Generate synthetic data
    data = torch.randint(0, vocab_size, (1000, seq_len))
    targets = torch.roll(data, -1, dims=1)
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_gate_rate = 0
        
        for batch_idx, (batch_data, batch_targets) in enumerate(dataloader):
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            output = model(batch_data, batch_targets)
            loss = output['loss']
            gate_rate = output['avg_gate_rate']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_gate_rate += gate_rate.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Gate Rate: {gate_rate.item():.3f}')
        
        avg_loss = total_loss / len(dataloader)
        avg_gate_rate = total_gate_rate / len(dataloader)
        print(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Avg Gate Rate: {avg_gate_rate:.3f}')
    
    print('Training completed!')
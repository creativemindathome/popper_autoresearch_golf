import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset

class FourierAttentionBlock(nn.Module):
    def __init__(self, d_model, num_freq_filters=64):
        super().__init__()
        self.d_model = d_model
        self.num_freq_filters = num_freq_filters
        
        # Frequency domain gates
        self.freq_gates = nn.Parameter(torch.randn(num_freq_filters, d_model) * 0.02)
        self.freq_bias = nn.Parameter(torch.zeros(num_freq_filters, d_model))
        
        # Mixing weights for combining frequency components
        self.freq_mix = nn.Linear(num_freq_filters, 1, bias=False)
        
        # Traditional components
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        residual = x
        x = self.norm1(x)
        
        # Apply FFT along sequence dimension
        x_freq = torch.fft.fft(x, dim=1)  # Complex tensor
        seq_len = x.size(1)
        
        # Take only meaningful frequency components
        n_freqs = min(self.num_freq_filters, seq_len // 2 + 1)
        x_freq_trunc = x_freq[:, :n_freqs, :]
        
        # Apply frequency-domain gates
        gates = torch.sigmoid(self.freq_gates[:n_freqs] + self.freq_bias[:n_freqs])
        x_gated = x_freq_trunc * gates.unsqueeze(0)
        
        # Reconstruct full frequency representation
        x_freq_full = torch.zeros_like(x_freq)
        x_freq_full[:, :n_freqs, :] = x_gated
        
        # Inverse FFT
        x_spatial = torch.fft.ifft(x_freq_full, dim=1).real
        
        # Mix frequency information
        freq_weights = F.softmax(self.freq_mix.weight[:n_freqs], dim=0)
        x_mixed = torch.sum(x_gated.real * freq_weights.view(1, -1, 1), dim=1, keepdim=True)
        x_mixed = x_mixed.expand(-1, seq_len, -1)
        
        # Combine spatial and frequency representations
        x = x_spatial + 0.1 * x_mixed
        x = x + residual
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x

class FourierGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, num_layers=6, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            FourierAttentionBlock(d_model) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
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
        pos_ids = torch.arange(seq_len, device=x.device)
        
        x = self.token_embedding(x) + self.pos_embedding(pos_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FourierGPT(vocab_size=1000, d_model=256, num_layers=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Generate synthetic data
batch_size = 8
seq_len = 128
num_batches = 100

# Create training data with some periodic patterns
train_data = []
for _ in range(num_batches):
    # Add some periodic structure to test frequency domain learning
    base = torch.randint(0, 1000, (batch_size, seq_len))
    # Add periodic pattern every 16 tokens
    for i in range(0, seq_len, 16):
        if i + 8 < seq_len:
            base[:, i:i+8] = torch.randint(100, 200, (batch_size, 8))
    train_data.append(base)

# Training loop
model.train()
train_losses = []

for epoch in range(20):
    epoch_losses = []
    
    for batch_idx, batch in enumerate(train_data):
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        optimizer.zero_grad()
        logits, loss = model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
        
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

print('Training completed!')
print(f'Final loss: {train_losses[-1]:.4f}')
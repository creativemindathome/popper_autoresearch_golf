import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MetabolicAttentionHead(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.energy = nn.Parameter(torch.tensor(1.0))
        self.age = 0
        self.split_threshold = 2.0
        self.merge_threshold = 0.3
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x) 
        v = self.v_proj(x)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        # Calculate metabolic energy based on gradient magnitude and attention entropy
        attn_entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean()
        self.energy.data = 0.9 * self.energy.data + 0.1 * attn_entropy
        self.age += 1
        
        return out, attn
    
    def should_split(self):
        return self.energy.item() > self.split_threshold and self.age > 10
    
    def should_merge(self):
        return self.energy.item() < self.merge_threshold and self.age > 20
    
    def split(self):
        # Create two new heads from this one
        head1 = MetabolicAttentionHead(self.d_model, self.head_dim)
        head2 = MetabolicAttentionHead(self.d_model, self.head_dim)
        
        # Initialize with slight variations of current weights
        noise_scale = 0.1
        with torch.no_grad():
            for param1, param2, orig_param in zip(head1.parameters(), head2.parameters(), self.parameters()):
                if param1.shape == orig_param.shape:
                    param1.data = orig_param.data + torch.randn_like(orig_param) * noise_scale
                    param2.data = orig_param.data - torch.randn_like(orig_param) * noise_scale
        
        return [head1, head2]

class MetabolicMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_heads=16):
        super().__init__()
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.max_heads = max_heads
        
        # Initialize with base number of heads
        self.heads = nn.ModuleList([
            MetabolicAttentionHead(d_model, self.head_dim) for _ in range(num_heads)
        ])
        
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        head_outputs = []
        attentions = []
        
        for head in self.heads:
            out, attn = head(x)
            head_outputs.append(out)
            attentions.append(attn)
        
        # Concatenate all head outputs
        if len(head_outputs) > 0:
            concat_out = torch.cat(head_outputs, dim=-1)
            # Adjust projection if we have different number of heads
            if concat_out.shape[-1] != self.d_model:
                # Adaptive projection
                proj_weight = self.out_proj.weight[:, :concat_out.shape[-1]]
                out = F.linear(concat_out, proj_weight, self.out_proj.bias)
            else:
                out = self.out_proj(concat_out)
        else:
            out = torch.zeros_like(x)
        
        return out
    
    def metabolic_update(self):
        """Perform metabolic operations: splitting, merging, pruning"""
        new_heads = []
        
        for head in self.heads:
            if head.should_split() and len(self.heads) < self.max_heads:
                # Split this head
                split_heads = head.split()
                new_heads.extend(split_heads)
            elif head.should_merge():
                # Skip this head (prune it)
                continue
            else:
                new_heads.append(head)
        
        # Update heads list
        self.heads = nn.ModuleList(new_heads)
        
        # Ensure we have at least one head
        if len(self.heads) == 0:
            self.heads = nn.ModuleList([MetabolicAttentionHead(self.d_model, self.head_dim)])

class MetabolicTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MetabolicMultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
    def metabolic_update(self):
        self.attn.metabolic_update()

class MetabolicGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([MetabolicTransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.metabolic_step = 0
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def metabolic_update(self):
        """Perform metabolic updates across all blocks"""
        self.metabolic_step += 1
        if self.metabolic_step % 50 == 0:  # Update every 50 steps
            for block in self.blocks:
                block.metabolic_update()
            print(f"Metabolic update at step {self.metabolic_step}")
            self.print_architecture()
    
    def print_architecture(self):
        for i, block in enumerate(self.blocks):
            num_heads = len(block.attn.heads)
            energies = [h.energy.item() for h in block.attn.heads]
            print(f"Layer {i}: {num_heads} heads, energies: {[f'{e:.3f}' for e in energies]}")

class SimpleDataset(Dataset):
    def __init__(self, vocab_size=1000, seq_len=64, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.cat([x[1:], torch.tensor([0])])
        return x, y

# Training setup
vocab_size = 1000
model = MetabolicGPT(vocab_size=vocab_size, d_model=256, num_heads=4, num_layers=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
dataset = SimpleDataset(vocab_size=vocab_size, num_samples=1000)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Starting training with Metabolic Growth Transformer")
model.print_architecture()

# Training loop
for epoch in range(3):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Perform metabolic update
        model.metabolic_update()
        
        total_loss += loss.item()
        
        if batch_idx % 20 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')

print("Final architecture:")
model.print_architecture()
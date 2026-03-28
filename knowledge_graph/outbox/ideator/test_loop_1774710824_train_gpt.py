
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hyperparameters:
    vocab_size = 50257
    d_model = 256
    n_heads = 4
    n_layers = 2
    d_mlp = 1024
    max_seq_len = 1024
    batch_size = 4
    learning_rate = 0.001

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_mlp, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids, targets=None):
        B, T = input_ids.size()
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

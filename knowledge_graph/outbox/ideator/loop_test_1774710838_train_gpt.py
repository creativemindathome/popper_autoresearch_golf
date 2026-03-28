import torch
import torch.nn as nn
import torch.nn.functional as F

class Hyperparameters:
    vocab_size = 50257
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_mlp = 512

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_mlp):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_mlp, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, targets=None):
        B, T = input_ids.size()
        x = self.tok_emb(input_ids) + self.pos_emb(torch.arange(T, device=input_ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits

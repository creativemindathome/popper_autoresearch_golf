import torch
import torch.nn as nn
import torch.nn.functional as F

class Hyperparameters:
    vocab_size = 64
    d_model = 32
    n_heads = 2
    n_layers = 1
    d_mlp = 64

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_mlp):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Initialize properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids, targets=None):
        x = self.tok_emb(input_ids)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        logits = self.lm_head(x)
        
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits

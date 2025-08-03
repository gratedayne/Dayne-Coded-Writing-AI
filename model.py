import torch
import torch.nn as nn

class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, ff_dim=512, max_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(max_len, embed_dim))
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        batch_size, seq_len = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding[:seq_len, :].unsqueeze(0)
        x = tok_emb + pos_emb
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
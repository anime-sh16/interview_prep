import torch.nn as nn

from transformer import MultiHeadAttention, FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        
        self.num_heads = num_heads
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask = None):
        x = self.dropout(self.self_attention(self.attn_norm(x), mask=mask)) + x
        x = self.dropout(self.feed_forward(self.ff_norm(x))) + x
        return x

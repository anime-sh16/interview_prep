import torch
import torch.nn as nn

from transformer import MultiHeadAttention, FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_x, causal_mask, decoder_padding_mask = None, encoder_padding_mask = None):
        
        # self attention
        self_mask = causal_mask if decoder_padding_mask is None else decoder_padding_mask & causal_mask
        x = self.dropout(self.self_attention(self.attn_norm(x), mask=self_mask)) + x
        
        # cross attention
        x = self.dropout(self.cross_attention(self.cross_attn_norm(x), mask=encoder_padding_mask, encoder_x=encoder_x)) + x
        
        # feed forward
        x = self.dropout(self.feed_forward(self.ff_norm(x))) + x
        return x
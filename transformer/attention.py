import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout: float = 0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`: {num_heads}).")
        
        self.dropout = nn.Dropout(dropout)
        
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        
        self.out = nn.Linear(embed_dim, embed_dim)
        

    def _project_and_reshape(self, matrix: nn.Linear, x):
        batch_size, seq_length, _ = x.size()
        res = matrix(x)
        return res.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, x, mask = None, encoder_x = None):
        batch_size, seq_length, _ = x.size()
        
        q = self._project_and_reshape(self.q, x) # (B, H, L, D)
        if encoder_x is not None:
            k = self._project_and_reshape(self.k, encoder_x) # (B, H, L_e, D)
            v = self._project_and_reshape(self.v, encoder_x) # (B, H, L_e, D)
        else:
            k = self._project_and_reshape(self.k, x)
            v = self._project_and_reshape(self.v, x)

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5 # (B, H, L, L_e)
        
        if mask is not None:
            attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn = torch.matmul(attn, v)
        
        output = attn.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out(output)
        
        return output


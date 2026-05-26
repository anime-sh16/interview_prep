import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        
        if d_ff is None:
            d_ff = d_model * 4
        
        # input: (B,L,D_m)
        self.layer = nn.Sequential(
            nn.Linear(d_model, d_ff), # (B,L,D_m) -> (B,L,D_ff)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), # (B,L,D_ff) -> (B,L,D_m)
        )
        
    def forward(self, x):
        return self.layer(x)
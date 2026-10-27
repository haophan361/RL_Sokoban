import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalModule(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 4, max_h: int = 20, max_w: int = 20):
        super().__init__()
        self.d_model = d_model

        self.row_embed = nn.Embedding(max_h, d_model // 2)
        self.col_embed = nn.Embedding(max_w, d_model // 2)
        self.pos_proj  = nn.Linear(d_model, d_model)

        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def _pos_encoding(self, H: int, W: int, device):
        h_emb = self.row_embed(torch.arange(H, device=device))  
        w_emb = self.col_embed(torch.arange(W, device=device)) 
        
        pos = torch.cat([
            h_emb.unsqueeze(1).expand(-1, W, -1), 
            w_emb.unsqueeze(0).expand(H, -1, -1), 
        ], dim=-1)

        pos = self.pos_proj(pos)                                
        return pos.reshape(1, H * W, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        entities = x.reshape(B, C, N).permute(0, 2, 1)

        entities = entities + self._pos_encoding(H, W, x.device)

        attn_out, _ = self.mha(entities, entities, entities)
        entities = self.norm1(entities + attn_out)

        ffn_out = self.ffn(entities)
        entities = self.norm2(entities + ffn_out)

        return entities.permute(0, 2, 1).reshape(B, C, H, W)

class AttentionPool(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        entities = x.reshape(B, C, H * W).permute(0, 2, 1)

        scores = self.attn(entities)
        weights = F.softmax(scores, dim=1)

        pooled = (entities * weights).sum(dim=1)
        return pooled


class DuelingDQN(nn.Module):

    def __init__(self, in_channels=4, d_model=128, num_heads=4):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.relational = RelationalModule(d_model, num_heads)

        self.attn_pool = AttentionPool(d_model)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Conv2d(d_model, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        h = self.conv1(x)           # (B, 64,     H, W)
        h = self.conv2(h)           # (B, d_model, H, W)
        res = h

        h = self.conv3(h)           # (B, d_model, H, W)
        h = h + res

        h = self.relational(h)      # (B, d_model, H, W)

        pooled = self.attn_pool(h)                                # (B, d_model)
        value = self.value_head(pooled)                           # (B, 1)

        advantage = self.advantage_stream(h)                      # (B, 4, H, W)
        adv_flat = advantage.permute(0, 2, 3, 1).contiguous().view(B, -1)  # (B, H*W*4)

        q = value + adv_flat - adv_flat.mean(dim=1, keepdim=True)
        return q
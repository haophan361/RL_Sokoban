import torch.nn as nn

class DuelingDQN(nn.Module):

    def __init__(self, in_channels=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, 128, H, W) → (B, 128, 1, 1)
            nn.Flatten(),             # (B, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),         # (B, 1) — scalar V(s)
        )

        self.advantage_stream = nn.Conv2d(128, 4, kernel_size=1)  # (B, 4, H, W)

    def forward(self, x):
        B = x.shape[0]
        features = self.features(x)                                    # (B, 128, H, W)

        value = self.value_stream(features)                            # (B, 1)
        advantage = self.advantage_stream(features)                    # (B, 4, H, W)

        # Flatten advantage: (B, H*W*4)
        adv_flat = advantage.permute(0, 2, 3, 1).contiguous().view(B, -1)

        q = value + adv_flat - adv_flat.mean(dim=1, keepdim=True)
        return q
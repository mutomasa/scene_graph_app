from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RelationCandidate:
    subject_index: int
    object_index: int
    spatial_features: Tuple[float, float, float, float]
    iou: float


class SimpleAttentionalGCN(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.rel_proj = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, relation_feats: torch.Tensor) -> torch.Tensor:
        # relation_feats: (N_pairs, in_dim)
        if relation_feats.numel() == 0:
            return torch.empty((0, 1), device=relation_feats.device)
        h = self.rel_proj(relation_feats)  # (N, hidden)
        # use self-attention over relation proposals as a GC-like refinement
        h_attn, _ = self.attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = 0.5 * h + 0.5 * h_attn.squeeze(0)
        score = torch.sigmoid(self.cls(h))  # (N, 1)
        return score


def refine_relations(candidates: List[RelationCandidate], device: str | None = None) -> List[float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not candidates:
        return []
    feats = torch.tensor([c.spatial_features for c in candidates], dtype=torch.float32, device=device)
    model = SimpleAttentionalGCN(in_dim=feats.shape[1]).to(device)
    model.eval()
    with torch.inference_mode():
        scores = model(feats).squeeze(-1).detach().cpu().tolist()
    return [float(s) for s in scores]



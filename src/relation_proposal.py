from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Pair:
    subject_index: int
    object_index: int
    spatial_features: Tuple[float, float, float, float]
    iou: float


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def extract_spatial_features(
    box_s: Tuple[float, float, float, float], box_o: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    sx1, sy1, sx2, sy2 = box_s
    ox1, oy1, ox2, oy2 = box_o
    sw = max(1e-5, sx2 - sx1)
    sh = max(1e-5, sy2 - sy1)
    sox = ((ox1 + ox2) / 2 - (sx1 + sx2) / 2) / sw
    soy = ((oy1 + oy2) / 2 - (sy1 + sy2) / 2) / sh
    srw = (ox2 - ox1) / sw
    srh = (oy2 - oy1) / sh
    return (sox, soy, srw, srh)


def relation_proposals(
    boxes: Iterable[Tuple[float, float, float, float]],
    max_pairs: int = 200,
    min_iou: float = 0.0,
) -> List[Pair]:
    boxes_list = list(boxes)
    num = len(boxes_list)
    pairs: List[Pair] = []
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            iou = compute_iou(boxes_list[i], boxes_list[j])
            if iou < min_iou:
                continue
            spatial = extract_spatial_features(boxes_list[i], boxes_list[j])
            pairs.append(Pair(i, j, spatial, iou))
    # score by proximity using L2 norm of spatial features (smaller is better)
    if not pairs:
        return []
    scores = np.array([np.linalg.norm(p.spatial_features) for p in pairs])
    order = np.argsort(scores)
    selected = [pairs[k] for k in order[: max_pairs]]
    return selected



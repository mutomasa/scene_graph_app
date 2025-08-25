from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    label: int
    score: float


class MaskRCNNDetector:
    def __init__(self, score_threshold: float = 0.5, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.score_threshold = score_threshold
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> List[Detection]:
        tensor = self.transform(image).to(self.device)
        outputs = self.model([tensor])[0]

        detections: List[Detection] = []
        boxes = outputs["boxes"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            if float(score) < self.score_threshold:
                continue
            x1, y1, x2, y2 = map(float, box.tolist())
            detections.append(Detection((x1, y1, x2, y2), int(label), float(score)))
        return detections


def get_coco_categories() -> List[str]:
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    categories = list(weights.meta.get("categories", []))
    return categories



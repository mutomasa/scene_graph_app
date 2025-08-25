from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from PIL import Image
import requests
import urllib3

from src.detector import MaskRCNNDetector, Detection, get_coco_categories
from src.relation_proposal import relation_proposals
from src.agcn_refine import RelationCandidate, refine_relations
from src.viz import build_graph, to_plotly_figure


@dataclass
class SceneGraph:
    boxes: List[Tuple[float, float, float, float]]
    labels: List[int]
    edges: List[Tuple[int, int, float]]  # (subject, object, score)


def run_pipeline(
    image: Image.Image,
    score_threshold: float = 0.6,
    edge_threshold: float = 0.1,
    top_k_fallback: int = 10,
) -> SceneGraph:
    detector = MaskRCNNDetector(score_threshold=score_threshold)
    detections: List[Detection] = detector.detect(image)
    boxes = [d.bbox for d in detections]
    labels = [d.label for d in detections]
    pairs = relation_proposals(boxes, max_pairs=200, min_iou=0.0)
    candidates = [
        RelationCandidate(p.subject_index, p.object_index, p.spatial_features, p.iou)
        for p in pairs
    ]
    scores = refine_relations(candidates)
    scored_edges = [
        (c.subject_index, c.object_index, float(s))
        for c, s in zip(candidates, scores)
    ]
    # しきい値で採択
    edges = [e for e in scored_edges if e[2] >= edge_threshold]
    # 全て落ちた場合は上位Kを表示（可視化のためのフォールバック）
    if not edges and scored_edges:
        scored_edges.sort(key=lambda x: x[2], reverse=True)
        edges = scored_edges[: top_k_fallback]
    return SceneGraph(boxes=boxes, labels=labels, edges=edges)


def main() -> None:
    st.set_page_config(page_title="Scene Graph Generator", page_icon="📈", layout="wide")
    st.title("Scene Graph Generator (Graph R-CNN inspired)")
    st.caption(
        "Mask R-CNN → Relation Proposals → Attentional refinement に基づくシーン・グラフ可視化"
    )
    st.write(
        "参考論文: Graph R-CNN for Scene Graph Generation (ECCV 2018)."
        " [arXiv:1808.00191](https://arxiv.org/abs/1808.00191)"
    )

    score_th = st.sidebar.slider("検出スコア閾値", 0.0, 1.0, 0.6, 0.05)
    rel_th = st.sidebar.slider("関係スコア閾値", 0.0, 1.0, 0.1, 0.05)
    uploaded = st.file_uploader("テスト画像をアップロード", type=["jpg", "jpeg", "png"])
    use_sample = st.button("サンプル画像を読み込む")

    col1, col2 = st.columns(2)
    img = None
    if uploaded is not None:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    elif use_sample:
        sample_urls = [
            # GitHub-hosted,安定性高
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
            "https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/astronaut.png",
            # COCO バックアップ
            "https://images.cocodataset.org/val2017/000000039769.jpg",
        ]
        last_err: Exception | None = None
        for url in sample_urls:
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                st.caption(f"サンプル画像: {url}")
                break
            except Exception as e:  # 最後に検証無効の最終手段
                last_err = e
                try:
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    resp = requests.get(url, timeout=20, verify=False)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    st.caption(f"サンプル画像(検証無効): {url}")
                    break
                except Exception as e2:
                    last_err = e2
                    continue
        if img is None and last_err is not None:
            st.error(f"サンプル画像の取得に失敗しました: {last_err}")

    if img is None:
        st.info("テスト画像をアップロードするか、サンプル画像を読み込んでください。")
        return

    col1.image(img, caption="入力画像", use_container_width=True)

    with st.spinner("推論中..."):
        sg = run_pipeline(img, score_threshold=score_th, edge_threshold=rel_th)

    g = build_graph(sg.boxes, sg.labels, sg.edges)
    categories = get_coco_categories()
    def label_name(idx: int) -> str:
        if 0 <= idx < len(categories):
            return categories[idx]
        return str(idx)
    node_text = [f"{i}: {label_name(l)}" for i, l in enumerate(sg.labels)]
    fig = to_plotly_figure(g, node_text=node_text)
    col2.plotly_chart(fig, use_container_width=True)

    st.subheader("エッジ一覧 (subject → object, score)")
    for s, o, sc in sg.edges:
        st.write(f"{s} → {o}: {sc:.3f}")


if __name__ == "__main__":
    main()



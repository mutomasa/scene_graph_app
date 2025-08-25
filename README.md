## Scene Graph App (Graph R-CNN inspired)

このプロジェクトは、画像からシーングラフを生成し、NetworkX/Plotly により可視化、Streamlit で表示する最小実装です。パイプラインは以下の 3 段階で構成されます。

1. Mask R-CNN による物体検出（Bounding Box）
2. Relation Proposal Network (RePN) 風の手法によりボックス組み合わせを抽出
3. attentional GCN 風のモジュールで関係スコアをリファイン

本実装の主要な貢献は 2 と 3 の簡易実装部分であり、ECCV 2018 の Graph R-CNN 論文に着想を得ています。

- 参考: Graph R-CNN for Scene Graph Generation (ECCV 2018) [arXiv:1808.00191](https://arxiv.org/abs/1808.00191)

### 技術的概要

- 検出: `torchvision` の `maskrcnn_resnet50_fpn` を推論モードで使用。
  - 出力の `boxes`, `labels`, `scores` からスコア閾値以上の検出を採用。
- Relation Proposal (RePN 簡易版): 全ペアを走査し、IoU と相対位置/サイズに基づく空間特徴 `(Δx, Δy, r_w, r_h)` を算出。L2 ノルムにより近接度でソートし上位を選択。
- attentional GCN 風リファイン: 空間特徴を線形射影し、Multi-Head Attention による自己注意で相互依存をモデリング。MLP により関係スコアを出力し、しきい値でエッジ採択。
- 可視化: `networkx.DiGraph` を構築し、`plotly` の `Scatter` でノード・エッジを描画。
- UI: Streamlit により画像アップロードとグラフ可視化を提供。

### セットアップ（uv パッケージマネージャ）

uv を使用して依存関係を解決します。

```bash
pip install uv
uv pip install -e .
```

GPU がある場合は PyTorch の適切なビルドをインストールしてください（CUDA バージョンに依存）。

### 実行方法

```bash
streamlit run app.py
```

ブラウザで表示される UI からテスト画像（jpg/png）をアップロードしてください。左に入力画像、右にシーングラフが表示され、下部に (subject → object, score) のエッジ一覧が表示されます。

### ディレクトリ構成

```
scene_graph_app/
  app.py                # Streamlit エントリーポイント
  src/
    detector.py         # Mask R-CNN 推論
    relation_proposal.py# RePN 簡易版でペア生成
    agcn_refine.py      # attentional 風 GCN でリファイン
    viz.py              # NetworkX/Plotly 可視化
  pyproject.toml        # uv/PEP 621 メタデータ
  README.md
```

### 注意事項

- 本実装は研究プロトタイプです。Graph R-CNN 論文の完全再現ではありませんが、主要な流れ（RePN と aGCN の思想）を簡易に体験できる形にしています。
- `maskrcnn_resnet50_fpn` の重みは自動でダウンロードされます。オフライン環境では事前にキャッシュが必要です。

### 引用

本実装を参照する場合は、以下の論文を引用してください。

Yang, J., Lu, J., Lee, S., Batra, D., & Parikh, D. Graph R-CNN for Scene Graph Generation. In ECCV 2018. [arXiv:1808.00191](https://arxiv.org/abs/1808.00191)



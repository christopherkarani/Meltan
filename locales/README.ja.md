# MetalANNS

**Apple Silicon ネイティブの GPU ベクトル検索。** Pure Swift + Metal。C++ 不要。クラウド不要。妥協なし。

実測値（FAISS / hnswlib / USearch / sqlite-vec との比較）は英語の [README](../README.md) と [BENCHMARKS.md](../BENCHMARKS.md)。デフォルト検索は exact です。

MetalANNS は、iOS、macOS、visionOS 向けのプロダクショングレードの近似最近傍探索ライブラリです。Metal コンピュートシェーダーによる GPU アクセラレーションで、完全にデバイス上で動作します。

*[English](../README.md) | [中文](README.zh-CN.md) | 日本語 | [Português](README.pt-BR.md) | [Español](README.es.md)*

## なぜ MetalANNS？

ほとんどの ANN ライブラリは、C++ を Apple プラットフォームに移植したものです。MetalANNS は、Metal のメモリモデルとコンピュートアーキテクチャのためにゼロから設計されました。

- **HNSW ではなく CAGRA** — 固定出次数の有向グラフにより、完全な GPU 並列化が可能。逐次挿入のボトルネックなし。HNSW と比較して[構築が 2.2〜27 倍高速、クエリが 33〜77 倍高速](https://arxiv.org/abs/2308.15136)。
- **デュアルバックエンド** — Apple Silicon では Metal シェーダー、シミュレータや CI では Accelerate（vDSP/BLAS）にフォールバック。同じ API、同じ結果。
- **ミュータブルインデックス** — 挿入、削除、バッチ更新、コンパクション。ビルドしたら検索のみではありません。
- **複数の永続化モード** — バイナリ保存/読み込み、ゼロコピー mmap、大規模インデックス向けのディスクバックドストリーミング。
- **型安全なフィルタリング** — ブール論理、範囲クエリ、集合メンバーシップをサポートするリッチなクエリ DSL。
- **Swift 6 コンカレンシー** — Actor ベースのスレッドセーフティ。コンパイル時に `Sendable` を強制。

## クイックスタート

```swift
// Package.swift
.package(url: "https://github.com/<your-org>/MetalANNS.git", from: "0.1.0")
```

```swift
import MetalANNS

// インデックスを構築
let index = VectorIndex<String, VectorIndexState.Unbuilt>(
    configuration: IndexConfiguration(degree: 32, metric: .cosine, efSearch: 64)
)

let ready = try await index.build(
    records: zip(ids, vectors).map { VectorRecord(id: $0.0, vector: $0.1) }
)

// フィルター付き検索
let results = try await ready.search(query: queryVector, topK: 10) {
    QueryFilter.equals(Field<String>("category"), "docs")
    QueryFilter.greaterThan(Field<Float>("score"), 0.8)
}

for hit in results {
    print("\(hit.id) -> \(hit.score)")
}
```

## ミュータビリティ + 永続化

```swift
// ライブ変更
try await index.insert(newVector, id: "doc_123")
try await index.batchInsert(batchVectors, ids: batchIDs)
try await index.delete(id: "doc_99")
try await index.compact()

// 保存と読み込み
try await index.save(to: fileURL)
let loaded = try await VectorIndex<String, VectorIndexState.Ready>.load(from: fileURL)

// 読み取り負荷の高いワークロード向けゼロコピー
let mmap = try await VectorIndex<String, VectorIndexState.ReadOnly>.loadReadOnly(from: fileURL, mode: .mmap)
```

## パフォーマンス

リポジトリ内のベンチマークハーネス（`swift run MetalANNSBenchmarks`）による合成データの結果：

| | Graph Index | IVFPQ |
|---|---|---|
| **Recall@10** | 1.000 | 0.406 - 0.997 |
| **スループット** | 57-102 QPS | 19-581 QPS |
| **トレードオフ** | 精度優先 | 速度調整可能 |

グラフインデックスはすべての `efSearch` 設定で**完全な再現率**を維持。IVFPQ は調整可能な速度/精度ノブを提供 — 再現率を犠牲にできる場合、スループットが最大 10 倍に。

## アーキテクチャ

```
MetalANNS（パブリック API）       MetalANNSCore（内部実装）
┌─────────────────────┐         ┌─────────────────────────────┐
│ VectorIndex<K,State> │────────▶│ NN-Descent グラフ構築         │
│ QueryFilter DSL      │         │ ビームサーチ（GPU + CPU）     │
│ 永続化レイヤー        │         │ Metal シェーダー / Accelerate │
│ メタデータ（GRDB）    │         │ FP16 / バイナリ / PQ コーデック│
└─────────────────────┘         │ バイナリ + mmap シリアライゼーション│
                                 └─────────────────────────────┘
```

**`VectorIndex<Key, State>`** — メイン API。型状態マシン：`Unbuilt` → `Ready` → `ReadOnly`。

**`Advanced.*`** — パワーユーザー向けの低レベルインデックス型への直接アクセス：

| 型 | ユースケース |
|---|---|
| `Advanced.GraphIndex` | 生の CAGRA スタイルグラフ |
| `Advanced.StreamingIndex` | バックグラウンドマージによる連続取り込み |
| `Advanced.ShardedIndex` | k-means ルーティングによる大規模データセット |
| `Advanced.IVFPQIndex` | 速度/メモリトレードオフのための直積量子化 |

## 距離メトリクス

`cosine` · `l2` · `innerProduct` · `hamming`

## ベンチマーク

```bash
swift run MetalANNSBenchmarks                        # ベースライン
swift run MetalANNSBenchmarks --sweep                 # efSearch スイープ
swift run MetalANNSBenchmarks --ivfpq                 # グラフ vs IVFPQ
swift run MetalANNSBenchmarks --dataset path/to/data  # 実データセット
swift run MetalANNSBenchmarks --csv-out results.csv   # エクスポート
```

## 動作要件

| プラットフォーム | 最小バージョン |
|---|---|
| macOS | 14+ |
| iOS | 17+ |
| visionOS | 1.0+ |

GPU アクセラレーションには Apple Silicon を推奨。Intel / シミュレータでは Accelerate にフォールバック。

## ライセンス

MIT

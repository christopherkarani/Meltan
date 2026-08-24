# MetalANNS

**Apple Silicon 原生 GPU 向量搜索。** 纯 Swift + Metal。无 C++。无云端。无妥协。

当前数字（对比 FAISS / hnswlib / USearch / sqlite-vec）见英文 [README](../README.md) 与 [BENCHMARKS.md](../BENCHMARKS.md)。默认搜索为精确扫描。

MetalANNS 为 iOS、macOS 和 visionOS 带来生产级近似最近邻搜索——完全在设备端运行，通过 Metal 计算着色器实现 GPU 加速。

*[English](../README.md) | 中文 | [日本語](README.ja.md) | [Português](README.pt-BR.md) | [Español](README.es.md)*

## 为什么选择 MetalANNS？

大多数 ANN 库只是将 C++ 移植到 Apple 平台。MetalANNS 从零开始，专为 Metal 的内存模型和计算架构而设计。

- **CAGRA，而非 HNSW** — 固定出度有向图完全可 GPU 并行化。无顺序插入瓶颈。相比 HNSW，[构建速度快 2.2-27 倍，查询速度快 33-77 倍](https://arxiv.org/abs/2308.15136)。
- **双后端** — Apple Silicon 使用 Metal 着色器，模拟器和 CI 使用 Accelerate（vDSP/BLAS）回退。相同 API，相同结果。
- **可变索引** — 插入、删除、批量更新、压缩。不仅仅是构建一次永久查询。
- **多种持久化模式** — 二进制保存/加载、零拷贝 mmap、面向大型索引的磁盘流式传输。
- **类型安全过滤** — 丰富的查询 DSL，支持布尔逻辑、范围查询和集合成员。
- **Swift 6 并发** — 基于 Actor 的线程安全，编译时强制 `Sendable`。

## 快速开始

```swift
// Package.swift
.package(url: "https://github.com/<your-org>/MetalANNS.git", from: "0.1.0")
```

```swift
import MetalANNS

// 构建索引
let index = VectorIndex<String, VectorIndexState.Unbuilt>(
    configuration: IndexConfiguration(degree: 32, metric: .cosine, efSearch: 64)
)

let ready = try await index.build(
    records: zip(ids, vectors).map { VectorRecord(id: $0.0, vector: $0.1) }
)

// 带过滤条件的搜索
let results = try await ready.search(query: queryVector, topK: 10) {
    QueryFilter.equals(Field<String>("category"), "docs")
    QueryFilter.greaterThan(Field<Float>("score"), 0.8)
}

for hit in results {
    print("\(hit.id) -> \(hit.score)")
}
```

## 可变性 + 持久化

```swift
// 实时变更
try await index.insert(newVector, id: "doc_123")
try await index.batchInsert(batchVectors, ids: batchIDs)
try await index.delete(id: "doc_99")
try await index.compact()

// 保存和加载
try await index.save(to: fileURL)
let loaded = try await VectorIndex<String, VectorIndexState.Ready>.load(from: fileURL)

// 零拷贝，适用于读密集型工作负载
let mmap = try await VectorIndex<String, VectorIndexState.ReadOnly>.loadReadOnly(from: fileURL, mode: .mmap)
```

## 性能

来自仓库内基准测试工具（`swift run MetalANNSBenchmarks`）的合成数据结果：

| | Graph Index | IVFPQ |
|---|---|---|
| **Recall@10** | 1.000 | 0.406 - 0.997 |
| **吞吐量** | 57-102 QPS | 19-581 QPS |
| **权衡** | 精度优先 | 速度可调 |

图索引在所有 `efSearch` 设置下保持**完美召回率**。IVFPQ 提供可调的速度/精度旋钮——在可以牺牲召回率时，吞吐量最高可提升 10 倍。

## 架构

```
MetalANNS（公共 API）             MetalANNSCore（内部实现）
┌─────────────────────┐         ┌─────────────────────────────┐
│ VectorIndex<K,State> │────────▶│ NN-Descent 图构建            │
│ QueryFilter DSL      │         │ 束搜索（GPU + CPU）          │
│ 持久化层              │         │ Metal 着色器 / Accelerate    │
│ 元数据（GRDB）        │         │ FP16 / 二进制 / PQ 编解码器  │
└─────────────────────┘         │ 二进制 + mmap 序列化         │
                                 └─────────────────────────────┘
```

**`VectorIndex<Key, State>`** — 主 API。类型状态机：`Unbuilt` → `Ready` → `ReadOnly`。

**`Advanced.*`** — 高级用户直接访问底层索引类型的接口：

| 类型 | 使用场景 |
|---|---|
| `Advanced.GraphIndex` | 原始 CAGRA 风格图 |
| `Advanced.StreamingIndex` | 后台合并的持续摄入 |
| `Advanced.ShardedIndex` | 使用 k-means 路由的大型数据集 |
| `Advanced.IVFPQIndex` | 乘积量化，用于速度/内存权衡 |

## 距离度量

`cosine` · `l2` · `innerProduct` · `hamming`

## 基准测试

```bash
swift run MetalANNSBenchmarks                        # 基线测试
swift run MetalANNSBenchmarks --sweep                 # efSearch 扫描
swift run MetalANNSBenchmarks --ivfpq                 # 图 vs IVFPQ
swift run MetalANNSBenchmarks --dataset path/to/data  # 真实数据集
swift run MetalANNSBenchmarks --csv-out results.csv   # 导出
```

## 系统要求

| 平台 | 最低版本 |
|---|---|
| macOS | 14+ |
| iOS | 17+ |
| visionOS | 1.0+ |

推荐 Apple Silicon 以获得 GPU 加速。在 Intel / 模拟器上回退到 Accelerate。

## 许可证

MIT

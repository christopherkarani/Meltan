# MetalANNS

**Busca vetorial nativa em GPU para Apple Silicon.** Swift + Metal puro. Sem C++. Sem nuvem. Sem compromisso.

Numeros atuais (tabela vs FAISS / hnswlib / USearch / sqlite-vec): [README em ingles](../README.md) e [BENCHMARKS.md](../BENCHMARKS.md). A busca padrao e exata.

MetalANNS traz busca aproximada de vizinhos mais proximos em nivel de producao para iOS, macOS e visionOS — rodando inteiramente no dispositivo com aceleracao GPU via Metal compute shaders.

*[English](../README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | Português | [Español](README.es.md)*

## Por que MetalANNS?

A maioria das bibliotecas ANN sao portes de C++ adaptados para plataformas Apple. MetalANNS foi projetado do zero para o modelo de memoria e arquitetura de computacao do Metal.

- **CAGRA, nao HNSW** — Grafos direcionados com grau de saida fixo sao totalmente paralelizaveis em GPU. Sem gargalo de insercao sequencial. [Construcao 2,2-27x mais rapida, consultas 33-77x mais rapidas](https://arxiv.org/abs/2308.15136) vs. HNSW.
- **Backend duplo** — Metal shaders no Apple Silicon, fallback para Accelerate (vDSP/BLAS) em simuladores e CI. Mesma API, mesmos resultados.
- **Indices mutaveis** — Inserir, deletar, atualizar em lote, compactar. Nao apenas construir-uma-vez-consultar-para-sempre.
- **Multiplos modos de persistencia** — Salvar/carregar binario, mmap zero-copy, streaming em disco para indices grandes.
- **Filtragem type-safe** — DSL de consulta rico com logica booleana, consultas por faixa e pertinencia a conjuntos.
- **Concorrencia Swift 6** — Seguranca de threads baseada em Actor com `Sendable` imposto em tempo de compilacao.

## Inicio Rapido

```swift
// Package.swift
.package(url: "https://github.com/<your-org>/MetalANNS.git", from: "0.1.0")
```

```swift
import MetalANNS

// Construir um indice
let index = VectorIndex<String, VectorIndexState.Unbuilt>(
    configuration: IndexConfiguration(degree: 32, metric: .cosine, efSearch: 64)
)

let ready = try await index.build(
    records: zip(ids, vectors).map { VectorRecord(id: $0.0, vector: $0.1) }
)

// Busca com filtros
let results = try await ready.search(query: queryVector, topK: 10) {
    QueryFilter.equals(Field<String>("category"), "docs")
    QueryFilter.greaterThan(Field<Float>("score"), 0.8)
}

for hit in results {
    print("\(hit.id) -> \(hit.score)")
}
```

## Mutabilidade + Persistencia

```swift
// Mutacoes em tempo real
try await index.insert(newVector, id: "doc_123")
try await index.batchInsert(batchVectors, ids: batchIDs)
try await index.delete(id: "doc_99")
try await index.compact()

// Salvar e carregar
try await index.save(to: fileURL)
let loaded = try await VectorIndex<String, VectorIndexState.Ready>.load(from: fileURL)

// Zero-copy para cargas de trabalho de leitura intensiva
let mmap = try await VectorIndex<String, VectorIndexState.ReadOnly>.loadReadOnly(from: fileURL, mode: .mmap)
```

## Desempenho

Benchmarks do harness incluso no repositorio (`swift run MetalANNSBenchmarks`), dados sinteticos:

| | Graph Index | IVFPQ |
|---|---|---|
| **Recall@10** | 1.000 | 0.406 - 0.997 |
| **Throughput** | 57-102 QPS | 19-581 QPS |
| **Trade-off** | Precisao primeiro | Velocidade ajustavel |

O indice de grafo mantem **recall perfeito** em todas as configuracoes de `efSearch`. IVFPQ oferece um controle ajustavel de velocidade/precisao — ate 10x mais throughput quando voce pode abrir mao de recall.

## Arquitetura

```
MetalANNS (API publica)          MetalANNSCore (internos)
┌─────────────────────┐         ┌─────────────────────────────┐
│ VectorIndex<K,State> │────────▶│ Construcao de grafo NN-Descent│
│ QueryFilter DSL      │         │ Beam search (GPU + CPU)     │
│ Camada de persistencia│        │ Metal shaders / Accelerate  │
│ Metadados (GRDB)     │         │ Codecs FP16 / Binario / PQ  │
└─────────────────────┘         │ Serializacao binaria + mmap  │
                                 └─────────────────────────────┘
```

**`VectorIndex<Key, State>`** — API principal. Maquina de estados tipada: `Unbuilt` → `Ready` → `ReadOnly`.

**`Advanced.*`** — Acesso direto a tipos de indice de baixo nivel para usuarios avancados:

| Tipo | Caso de Uso |
|---|---|
| `Advanced.GraphIndex` | Grafo estilo CAGRA bruto |
| `Advanced.StreamingIndex` | Ingestao continua com merges em background |
| `Advanced.ShardedIndex` | Grandes datasets com roteamento k-means |
| `Advanced.IVFPQIndex` | Quantizacao de produto para trade-off velocidade/memoria |

## Metricas de Distancia

`cosine` · `l2` · `innerProduct` · `hamming`

## Benchmarks

```bash
swift run MetalANNSBenchmarks                        # baseline
swift run MetalANNSBenchmarks --sweep                 # sweep de efSearch
swift run MetalANNSBenchmarks --ivfpq                 # grafo vs IVFPQ
swift run MetalANNSBenchmarks --dataset path/to/data  # dataset real
swift run MetalANNSBenchmarks --csv-out results.csv   # exportar
```

## Requisitos

| Plataforma | Minimo |
|---|---|
| macOS | 14+ |
| iOS | 17+ |
| visionOS | 1.0+ |

Apple Silicon recomendado para aceleracao GPU. Fallback para Accelerate em Intel / simuladores.

## Licenca

MIT

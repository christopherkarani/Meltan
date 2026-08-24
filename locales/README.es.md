# MetalANNS

**Busqueda vectorial nativa en GPU para Apple Silicon.** Swift + Metal puro. Sin C++. Sin nube. Sin compromisos.

Numeros actuales (tabla vs FAISS / hnswlib / USearch / sqlite-vec): [README en ingles](../README.md) y [BENCHMARKS.md](../BENCHMARKS.md). La busqueda por defecto es exacta.

MetalANNS trae busqueda aproximada de vecinos mas cercanos de nivel produccion a iOS, macOS y visionOS — ejecutandose completamente en el dispositivo con aceleracion GPU mediante Metal compute shaders.

*[English](../README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Português](README.pt-BR.md) | Español*

## Por que MetalANNS?

La mayoria de las bibliotecas ANN son ports de C++ adaptados a plataformas Apple. MetalANNS fue disenado desde cero para el modelo de memoria y la arquitectura de computo de Metal.

- **CAGRA, no HNSW** — Los grafos dirigidos con grado de salida fijo son completamente paralelizables en GPU. Sin cuello de botella de insercion secuencial. [Construccion 2.2-27x mas rapida, consultas 33-77x mas rapidas](https://arxiv.org/abs/2308.15136) vs. HNSW.
- **Backend dual** — Metal shaders en Apple Silicon, fallback a Accelerate (vDSP/BLAS) en simuladores y CI. Misma API, mismos resultados.
- **Indices mutables** — Insertar, eliminar, actualizar por lotes, compactar. No solo construir-una-vez-consultar-siempre.
- **Multiples modos de persistencia** — Guardado/carga binario, mmap zero-copy, streaming en disco para indices grandes.
- **Filtrado type-safe** — DSL de consulta rico con logica booleana, consultas por rango y pertenencia a conjuntos.
- **Concurrencia Swift 6** — Seguridad de hilos basada en Actor con `Sendable` impuesto en tiempo de compilacion.

## Inicio Rapido

```swift
// Package.swift
.package(url: "https://github.com/<your-org>/MetalANNS.git", from: "0.1.0")
```

```swift
import MetalANNS

// Construir un indice
let index = VectorIndex<String, VectorIndexState.Unbuilt>(
    configuration: IndexConfiguration(degree: 32, metric: .cosine, efSearch: 64)
)

let ready = try await index.build(
    records: zip(ids, vectors).map { VectorRecord(id: $0.0, vector: $0.1) }
)

// Busqueda con filtros
let results = try await ready.search(query: queryVector, topK: 10) {
    QueryFilter.equals(Field<String>("category"), "docs")
    QueryFilter.greaterThan(Field<Float>("score"), 0.8)
}

for hit in results {
    print("\(hit.id) -> \(hit.score)")
}
```

## Mutabilidad + Persistencia

```swift
// Mutaciones en vivo
try await index.insert(newVector, id: "doc_123")
try await index.batchInsert(batchVectors, ids: batchIDs)
try await index.delete(id: "doc_99")
try await index.compact()

// Guardar y cargar
try await index.save(to: fileURL)
let loaded = try await VectorIndex<String, VectorIndexState.Ready>.load(from: fileURL)

// Zero-copy para cargas de trabajo de lectura intensiva
let mmap = try await VectorIndex<String, VectorIndexState.ReadOnly>.loadReadOnly(from: fileURL, mode: .mmap)
```

## Rendimiento

Benchmarks del harness incluido en el repositorio (`swift run MetalANNSBenchmarks`), datos sinteticos:

| | Graph Index | IVFPQ |
|---|---|---|
| **Recall@10** | 1.000 | 0.406 - 0.997 |
| **Throughput** | 57-102 QPS | 19-581 QPS |
| **Trade-off** | Precision primero | Velocidad ajustable |

El indice de grafo mantiene **recall perfecto** en todas las configuraciones de `efSearch`. IVFPQ ofrece un control ajustable de velocidad/precision — hasta 10x mas throughput cuando se puede sacrificar recall.

## Arquitectura

```
MetalANNS (API publica)          MetalANNSCore (internos)
┌─────────────────────┐         ┌─────────────────────────────┐
│ VectorIndex<K,State> │────────▶│ Construccion de grafo NN-Descent│
│ QueryFilter DSL      │         │ Beam search (GPU + CPU)     │
│ Capa de persistencia │         │ Metal shaders / Accelerate  │
│ Metadatos (GRDB)     │         │ Codecs FP16 / Binario / PQ  │
└─────────────────────┘         │ Serializacion binaria + mmap │
                                 └─────────────────────────────┘
```

**`VectorIndex<Key, State>`** — API principal. Maquina de estados tipada: `Unbuilt` → `Ready` → `ReadOnly`.

**`Advanced.*`** — Acceso directo a tipos de indice de bajo nivel para usuarios avanzados:

| Tipo | Caso de Uso |
|---|---|
| `Advanced.GraphIndex` | Grafo estilo CAGRA sin procesar |
| `Advanced.StreamingIndex` | Ingesta continua con merges en segundo plano |
| `Advanced.ShardedIndex` | Grandes datasets con enrutamiento k-means |
| `Advanced.IVFPQIndex` | Cuantizacion de producto para trade-off velocidad/memoria |

## Metricas de Distancia

`cosine` · `l2` · `innerProduct` · `hamming`

## Benchmarks

```bash
swift run MetalANNSBenchmarks                        # linea base
swift run MetalANNSBenchmarks --sweep                 # barrido de efSearch
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

Apple Silicon recomendado para aceleracion GPU. Fallback a Accelerate en Intel / simuladores.

## Licencia

MIT
